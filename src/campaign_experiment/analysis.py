"""Run the pre-registered analysis end to end and collect every number.

The order mirrors docs/TEST_PLAN.md: validity checks first, then the
primary test, robustness, secondary metrics, variance reduction, segments,
economics and the decision. Every figure in the report and the memo is
read from the dictionary this module returns; nothing is typed by hand.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd

from campaign_experiment import __version__
from campaign_experiment.checks import (
    CATEGORICAL_COVARIATES,
    NUMERIC_COVARIATES,
    covariate_balance,
    srm_check,
)
from campaign_experiment.config import Config
from campaign_experiment.decision import ArmDecision, decide
from campaign_experiment.economics import campaign_economics
from campaign_experiment.logger import get_logger
from campaign_experiment.metrics_sql import arm_metrics, connect, segment_metrics
from campaign_experiment.peeking import simulate_peeking
from campaign_experiment.power import achieved_power_mean, mde_mean, mde_proportion
from campaign_experiment.segments import segment_effects
from campaign_experiment.stats import (
    bootstrap_diff_ci,
    holm,
    two_proportion_ztest,
    welch_test,
    winsorize,
)
from campaign_experiment.uplift import fit_t_learner
from campaign_experiment.variance_reduction import cuped_effect, regression_adjusted_effect

log = get_logger(__name__)


def _welch_from_rows(yc: np.ndarray, yt: np.ndarray, alpha: float, metric: str):
    return welch_test(
        float(yc.mean()),
        float(yc.var(ddof=1)),
        len(yc),
        float(yt.mean()),
        float(yt.var(ddof=1)),
        len(yt),
        alpha,
        metric,
    )


def _thin(curve: pd.DataFrame, points: int) -> pd.DataFrame:
    """Keep about `points` rows for storage, always including the first and last."""
    step = max(1, len(curve) // points)
    keep = sorted(set(range(0, len(curve), step)) | {len(curve) - 1})
    return curve.iloc[keep]


def _winsorized_check(
    spend: dict[str, np.ndarray],
    control: str,
    treatments: list[str],
    positive: pd.Series,
    quantile: float,
    alpha: float,
) -> dict[str, Any]:
    """Welch tests on spend capped at a pooled quantile of positive spend, Holm-corrected."""
    cap = float(positive.quantile(quantile))
    results = []
    for t_arm in treatments:
        est = _welch_from_rows(
            winsorize(spend[control], cap), winsorize(spend[t_arm], cap), alpha, "spend_winsorized"
        )
        results.append({"arm": t_arm, **est.to_dict()})
    for row, p_adj in zip(results, holm([r["p_value"] for r in results]), strict=True):
        row["p_holm"] = p_adj
        row["significant"] = p_adj < alpha
    return {"cap": cap, "quantile": quantile, "results": results}


def run_analysis(cfg: Config, df: pd.DataFrame) -> dict[str, Any]:
    exp = cfg.experiment
    control, treatments, alpha = exp.control_arm, exp.treatment_arms, exp.alpha
    con = connect(df)
    arms = arm_metrics(con).reindex(exp.arms)
    spend = {arm: df.loc[df["segment"] == arm, "spend"].to_numpy(float) for arm in exp.arms}

    # 1. Validity: sample ratio and covariate balance, before any effect.
    srm = srm_check(arms["n"].astype(int).to_dict(), exp.expected_allocation, exp.srm_alpha)
    balance = covariate_balance(df, control, treatments, exp.balance_smd_threshold)
    validity_passed = srm.passed
    log.info("SRM p=%.4f (%s)", srm.p_value, "pass" if srm.passed else "FAIL")

    # 2. Primary: revenue per customer, Welch from the SQL aggregates,
    #    bootstrap CI from rows, Holm across the two treatment comparisons.
    c = arms.loc[control]
    primary = []
    for t_arm in treatments:
        t = arms.loc[t_arm]
        est = welch_test(
            c["revenue_per_customer"],
            c["revenue_var"],
            int(c["n"]),
            t["revenue_per_customer"],
            t["revenue_var"],
            int(t["n"]),
            alpha,
            "spend",
        )
        low, high, _ = bootstrap_diff_ci(
            spend[control],
            spend[t_arm],
            cfg.bootstrap.n_resamples,
            cfg.bootstrap.confidence,
            cfg.bootstrap.seed,
        )
        primary.append({"arm": t_arm, **est.to_dict(), "boot_ci_low": low, "boot_ci_high": high})
    for row, p_adj in zip(primary, holm([r["p_value"] for r in primary]), strict=True):
        row["p_holm"] = p_adj
        row["significant"] = p_adj < alpha

    # 3. Robustness: cap at the pooled quantile of positive spend (plan, section 3).
    positive = df.loc[df["spend"] > 0, "spend"]
    winsorized = _winsorized_check(
        spend, control, treatments, positive, exp.winsorize_quantile, alpha
    )
    posthoc_winsorized = _winsorized_check(
        spend, control, treatments, positive, cfg.posthoc.winsorize_quantile, alpha
    )
    max_spend = float(positive.max())
    top_coding = {
        "max_spend": max_spend,
        "purchases_at_max": int((positive == max_spend).sum()),
        "purchases": len(positive),
        # The planned cap only trims anything if it sits below the maximum.
        "planned_cap_binding": winsorized["cap"] < max_spend,
    }

    # 4. Secondary metrics: rates, Holm across all four comparisons.
    secondary = []
    for metric, count_col in (("conversion", "conversions"), ("visit", "visits")):
        for t_arm in treatments:
            t = arms.loc[t_arm]
            est = two_proportion_ztest(
                int(c[count_col]), int(c["n"]), int(t[count_col]), int(t["n"]), alpha, metric
            )
            secondary.append({"arm": t_arm, **est.to_dict()})
    for row, p_adj in zip(secondary, holm([r["p_value"] for r in secondary]), strict=True):
        row["p_holm"] = p_adj
        row["significant"] = p_adj < alpha

    # 5. Head to head: is the choice between the two emails itself supported?
    a, b = treatments
    h2h = _welch_from_rows(spend[a], spend[b], alpha, "spend").to_dict()
    h2h.update({"baseline_arm": a, "compared_arm": b})

    # 6. Variance reduction (reported, not used for the decision).
    variance_reduction = []
    for t_arm in treatments:
        cuped = cuped_effect(df, "spend", exp.cuped_covariate, control, t_arm, alpha)
        reg = regression_adjusted_effect(
            df, "spend", control, t_arm, NUMERIC_COVARIATES, CATEGORICAL_COVARIATES, alpha
        )
        unadjusted_se = next(r["se"] for r in primary if r["arm"] == t_arm)
        variance_reduction.append(
            {
                "arm": t_arm,
                "cuped": {**cuped.estimate.to_dict(), "theta": cuped.theta},
                "cuped_variance_reduction": cuped.variance_reduction,
                "cuped_correlation": cuped.correlation,
                "regression": reg.to_dict(),
                "regression_se_reduction": 1 - reg.se / unadjusted_se,
            }
        )

    # 7. What could this test detect? MDE at the observed control baseline.
    n_c = int(c["n"])
    n_t_min = int(arms.loc[treatments, "n"].min())
    planning_alpha = alpha / len(treatments)
    sd_c = float(np.sqrt(c["revenue_var"]))
    observed_power = {
        "alpha_per_comparison": planning_alpha,
        "power": exp.power,
        "spend_sd_control": sd_c,
        "mde_spend": mde_mean(sd_c, n_c, n_t_min, planning_alpha, exp.power),
        "mde_conversion": mde_proportion(
            float(c["conversion_rate"]), n_c, n_t_min, planning_alpha, exp.power
        ),
        "mde_visit": mde_proportion(
            float(c["visit_rate"]), n_c, n_t_min, planning_alpha, exp.power
        ),
        "power_at_observed_effect": {
            r["arm"]: achieved_power_mean(r["diff"], sd_c, n_c, r["n_treatment"], planning_alpha)
            for r in primary
        },
    }

    # 8. Segments: Holm across every level of every segment, per arm.
    seg = segment_metrics(con, exp.segments)
    segments = {t_arm: segment_effects(seg, control, t_arm, alpha) for t_arm in treatments}

    # 9. Economics and the decision rule.
    economics, decisions = {}, []
    for row, wins in zip(primary, winsorized["results"], strict=True):
        t_arm = row["arm"]
        econ = campaign_economics(
            row["diff"],
            (row["boot_ci_low"], row["boot_ci_high"]),
            cfg.economics.gross_margin,
            cfg.economics.cost_per_email,
            cfg.economics.rollout_customers,
        )
        economics[t_arm] = econ.to_dict()
        positive_primary = row["significant"] and row["diff"] > 0
        decisions.append(
            ArmDecision(
                arm=t_arm,
                validity_passed=validity_passed,
                primary_significant_positive=positive_primary,
                winsorized_agrees=bool(
                    wins["significant"] == row["significant"]
                    and np.sign(wins["diff"]) == np.sign(row["diff"])
                ),
                no_segment_harm=not bool(segments[t_arm]["harm"].any()),
                profitable=econ.net_profit_per_customer > 0,
            )
        )
    recommended = decide(decisions, {a: e["net_profit_per_customer"] for a, e in economics.items()})
    log.info("recommendation: %s", recommended or "send neither email")

    # 10. Illustrations outside the decision: peeking and uplift targeting.
    pk = cfg.peeking
    peeking = simulate_peeking(
        float(c["conversion_rate"]), pk.n_per_arm, pk.n_looks, pk.n_simulations, alpha, pk.seed
    )
    up = cfg.uplift
    uplift = fit_t_learner(
        df, control, up.treatment_arm, up.outcome, up.test_size, up.seed, up.n_bins
    )

    return {
        "version": __version__,
        "dataset": {
            "rows": len(df),
            "arms": exp.arms,
            "duplicate_rows_kept": int(df.drop(columns=["arm"]).duplicated().sum()),
        },
        "arm_metrics": arms.reset_index().to_dict(orient="records"),
        "validity": {
            "srm": srm.to_dict(),
            "balance": balance.to_dict(orient="records"),
            "max_abs_smd": float(balance["smd"].abs().max()),
            "all_balanced": bool(balance["balanced"].all()),
            "smd_threshold": exp.balance_smd_threshold,
        },
        "primary": primary,
        "winsorized": winsorized,
        "top_coding": top_coding,
        "posthoc_winsorized": posthoc_winsorized,
        "secondary": secondary,
        "head_to_head": h2h,
        "variance_reduction": variance_reduction,
        "observed_power": observed_power,
        "segments": {arm: s.to_dict(orient="records") for arm, s in segments.items()},
        "economics": economics,
        "assumptions": asdict(cfg.economics),
        "decision": {
            "arms": [d.to_dict() for d in decisions],
            "recommended_arm": recommended,
        },
        "peeking": peeking.to_dict(),
        "uplift": {
            "treatment_arm": uplift.treatment_arm,
            "outcome": uplift.outcome,
            "n_train": uplift.n_train,
            "n_test": uplift.n_test,
            "qini_coefficient": uplift.qini_coefficient,
            "top_fraction": uplift.top_fraction,
            "top_share_of_incremental": uplift.top_share_of_incremental,
            "deciles": uplift.deciles.to_dict(orient="records"),
            "qini_curve": _thin(uplift.qini_curve, 200).to_dict(orient="records"),
        },
    }
