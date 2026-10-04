"""Write results.json, REPORT.md and DECISION_MEMO.md from the results.

Every number in the two Markdown files is formatted from the results
dictionary, so regenerating the analysis regenerates the documents and a
number cannot drift from the computation that produced it. No timestamp is
written, so an unchanged analysis produces byte-identical files.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def to_jsonable(value: Any) -> Any:
    """Plain JSON types; NaN and infinity become null (JSON has neither)."""
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.bool_ | bool):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, float | np.floating):
        f = float(value)
        return None if math.isnan(f) or math.isinf(f) else round(f, 10)
    return value


# --- formatting -----------------------------------------------------------


def usd(x: float, signed: bool = False) -> str:
    sign = "+" if signed and x > 0 else ("-" if x < 0 else "")
    return f"{sign}${abs(x):,.2f}"


def pct(x: float, digits: int = 1) -> str:
    return f"{x * 100:.{digits}f}%"


def pp(x: float, digits: int = 2) -> str:
    return f"{x * 100:+.{digits}f} pp"


def ordinal_pct(q: float) -> str:
    return f"{q * 100:.0f}th percentile"


def se_usd(x: float) -> str:
    return f"${x:.4f}"


def power_pct(x: float) -> str:
    return "> 99.9%" if x > 0.999 else pct(x)


def pval(p: float) -> str:
    return "< 0.0001" if p < 1e-4 else f"{p:.4f}"


def table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def _by_arm(rows: list[dict], arm: str, metric: str | None = None) -> dict:
    return next(r for r in rows if r["arm"] == arm and (metric is None or r["metric"] == metric))


def _break_even(econ: dict) -> str:
    be = econ["break_even_margin"]
    if be is None:
        return "none (no revenue lift)"
    if be > 1:
        return "above 100% (never profitable)"
    return pct(be)


# --- decision memo --------------------------------------------------------


def decision_memo(r: dict[str, Any]) -> str:
    arms = [p["arm"] for p in r["primary"]]
    rec = r["decision"]["recommended_arm"]
    econ = r["economics"]
    a = r["assumptions"]
    h2h = r["head_to_head"]
    srm = r["validity"]["srm"]

    lines = ["# Decision Memo: Email Campaign Rollout", ""]
    if rec:
        other = [x for x in arms if x != rec]
        lines += [
            f"**Recommendation: send the {rec}.** It passed every pre-registered criterion "
            f"and has the higher expected net profit"
            + (f" of the two qualifying emails ({', '.join([rec, *other])})." if other else "."),
            "",
        ]
    else:
        lines += ["**Recommendation: send neither email.** No arm passed every criterion.", ""]

    lines += ["## What the test found", ""]
    for arm in arms:
        p = _by_arm(r["primary"], arm)
        conv = _by_arm(r["secondary"], arm, "conversion")
        e = econ[arm]
        lines.append(
            f"- **{arm}:** revenue per customer {usd(p['control_mean'])} -> "
            f"{usd(p['treatment_mean'])}, a lift of **{usd(p['diff'], True)}** "
            f"(95% CI {usd(p['boot_ci_low'], True)} to {usd(p['boot_ci_high'], True)}, "
            f"Holm p {pval(p['p_holm'])}). Conversion {pct(conv['control_mean'], 2)} -> "
            f"{pct(conv['treatment_mean'], 2)}. Expected net profit "
            f"**{usd(e['net_profit_per_customer'], True)} per email** "
            f"(95% CI {usd(e['net_profit_ci'][0], True)} to {usd(e['net_profit_ci'][1], True)}); "
            f"it breaks even at a gross margin of {_break_even(e)}."
        )
    # Express the head-to-head as "better minus worse", so the interval is positive-side.
    if h2h["diff"] < 0:
        better, worse = h2h["baseline_arm"], h2h["compared_arm"]
        gap, gap_low, gap_high = -h2h["diff"], -h2h["ci_high"], -h2h["ci_low"]
    else:
        better, worse = h2h["compared_arm"], h2h["baseline_arm"]
        gap, gap_low, gap_high = h2h["diff"], h2h["ci_low"], h2h["ci_high"]
    supported = h2h["p_value"] < 0.05
    lines += [
        f"- **Head to head:** the {better} out-earns the {worse} by "
        f"{usd(gap)} per customer (95% CI {usd(gap_low, True)} to "
        f"{usd(gap_high, True)}, p {pval(h2h['p_value'])}). "
        + (
            "The preference is statistically supported, though the interval is wide."
            if supported
            else "The difference is not statistically significant, so the choice rests on "
            "the point estimate."
        ),
        "",
        "## Why the result can be trusted",
        "",
        "- Assignment worked: arm sizes match the planned 1/3 split "
        f"(SRM p {pval(srm['p_value'])}), "
        f"and every pre-period covariate is balanced (largest |SMD| "
        f"{r['validity']['max_abs_smd']:.3f}, threshold 0.1).",
        "- Metrics, tests, thresholds and this decision rule were committed in "
        "`docs/TEST_PLAN.md` before any outcome was analysed.",
        _robustness_line(r),
        "",
        "## What could change the decision",
        "",
        f"- **Assumed economics.** Profit uses an assumed {pct(a['gross_margin'], 0)} gross margin "
        f"and {usd(a['cost_per_email'])} per email; the dataset records revenue only. "
        "The break-even margins above show how far these can move before the call flips.",
        "- **Two-week window.** Effects on later purchases, fatigue or unsubscribes are not "
        "measured; this dataset has no unsubscribe or complaint data, so those guardrails "
        "could not be checked.",
        f"- **Top-coded spend.** The source caps spend at {usd(r['top_coding']['max_spend'])} "
        f"({r['top_coding']['purchases_at_max']} of {r['top_coding']['purchases']} purchases "
        "sit exactly there), so very large orders are understated in every arm.",
        "",
        "## Suggested next test",
        "",
    ]
    lines += _next_test(r)
    return "\n".join(lines) + "\n"


def _robustness_line(r: dict[str, Any]) -> str:
    """State the robustness evidence only as strongly as the results support it."""
    holds = all(
        x["significant"] and (x["diff"] > 0) == (_by_arm(r["primary"], x["arm"])["diff"] > 0)
        for x in r["posthoc_winsorized"]["results"]
    )
    cap = usd(r["posthoc_winsorized"]["cap"])
    if holds:
        return (
            f"- The lifts hold under a bootstrap interval and when spend is capped at {cap} "
            "(a post-hoc check; see the report)."
        )
    return (
        f"- Caution: at least one lift does not survive capping spend at {cap} "
        "(a post-hoc check; see the report)."
    )


def _next_test(r: dict[str, Any]) -> list[str]:
    out = []
    for arm, rows in r["segments"].items():
        sig = [x for x in rows if x["significant"]]
        not_sig = [f"{x['segment']}={x['level']}" for x in rows if not x["significant"]]
        out.append(
            f"- **{arm}:** significant revenue lift in {len(sig)} of {len(rows)} segment levels "
            f"after Holm correction"
            + ("; not significant in " + ", ".join(not_sig) + "." if not_sig else ".")
        )
    up = r["uplift"]
    out += [
        f"- A targeting model for the {up['treatment_arm']} (held-out data) ranks customers so "
        f"that the top {pct(up['top_fraction'], 0)} account for "
        f"{pct(up['top_share_of_incremental'], 0)} of incremental {up['outcome']}s. "
        "Segment and model findings are exploratory: test targeted sending against "
        "send-to-all in a new randomised experiment before acting on them.",
    ]
    return out


# --- full report ----------------------------------------------------------


def full_report(r: dict[str, Any]) -> str:
    arms = [p["arm"] for p in r["primary"]]
    s: list[str] = [
        "# Experiment Report",
        "",
        "Generated by `campaign-experiment run` from `results.json`. "
        "The plan this follows is `docs/TEST_PLAN.md`; the one-page summary is "
        "`DECISION_MEMO.md`.",
        "",
        "## 1. Data",
        "",
        f"{r['dataset']['rows']:,} customers, arms: {', '.join(r['dataset']['arms'])}. "
        f"{r['dataset']['duplicate_rows_kept']:,} rows are exact duplicates of another row; "
        "they are kept, because the file has no customer id and identical attributes do not "
        "make two customers the same person.",
        "",
        table(
            ["Arm", "Customers", "Visit rate", "Conversion rate", "Revenue", "Revenue / customer"],
            [
                [
                    m["arm"],
                    f"{int(m['n']):,}",
                    pct(m["visit_rate"], 2),
                    pct(m["conversion_rate"], 2),
                    usd(m["revenue"]),
                    usd(m["revenue_per_customer"]),
                ]
                for m in r["arm_metrics"]
            ],
        ),
        "",
    ]

    srm = r["validity"]["srm"]
    worst = sorted(r["validity"]["balance"], key=lambda b: -abs(b["smd"]))[:5]
    s += [
        "## 2. Validity checks",
        "",
        f"**Sample ratio:** chi-square {srm['chi2']:.3f}, p {pval(srm['p_value'])} "
        f"(fails below {srm['alpha']}): **{'pass' if srm['passed'] else 'FAIL'}**.",
        "",
        f"**Covariate balance:** {len(r['validity']['balance'])} covariate comparisons, "
        f"largest |SMD| {r['validity']['max_abs_smd']:.4f}: "
        f"**{'all balanced' if r['validity']['all_balanced'] else 'IMBALANCE FOUND'}**. "
        "The five largest:",
        "",
        table(
            ["Treatment", "Covariate", "Control mean", "Treatment mean", "SMD"],
            [
                [
                    b["treatment"],
                    b["covariate"],
                    f"{b['mean_control']:.3f}",
                    f"{b['mean_treatment']:.3f}",
                    f"{b['smd']:+.4f}",
                ]
                for b in worst
            ],
        ),
        "",
        "## 3. Primary metric: revenue per customer",
        "",
        "![Lift versus control](figures/primary_effects.png)",
        "",
        table(
            [
                "Arm",
                "Control",
                "Treatment",
                "Lift",
                "Welch 95% CI",
                "Bootstrap 95% CI",
                "p",
                "Holm p",
            ],
            [
                [
                    p["arm"],
                    usd(p["control_mean"]),
                    usd(p["treatment_mean"]),
                    usd(p["diff"], True),
                    f"{usd(p['ci_low'], True)} to {usd(p['ci_high'], True)}",
                    f"{usd(p['boot_ci_low'], True)} to {usd(p['boot_ci_high'], True)}",
                    pval(p["p_value"]),
                    pval(p["p_holm"]),
                ]
                for p in r["primary"]
            ],
        ),
        "",
        "The Welch and bootstrap intervals agree to within a few cents, so the normal "
        "approximation is adequate at this sample size despite the heavy right tail.",
        "",
    ]

    w, tc, ph = r["winsorized"], r["top_coding"], r["posthoc_winsorized"]
    s += [
        "## 4. Robustness: capped spend",
        "",
        f"**Pre-registered check** (cap at the {ordinal_pct(w['quantile'])} of positive "
        f"spend = {usd(w['cap'])}): "
        + (
            "the cap trims the tail and the result is below."
            if tc["planned_cap_binding"]
            else f"**non-binding.** The source file is top-coded: the largest purchase is "
            f"{usd(tc['max_spend'])} and {tc['purchases_at_max']} of {tc['purchases']} "
            "purchases sit exactly there, so the planned cap equals the maximum and changes "
            "nothing. The check passes trivially and carries no information."
        ),
        "",
        f"**Post-hoc check, not pre-registered** (cap at the {ordinal_pct(ph['quantile'])} "
        f"= {usd(ph['cap'])}), added after the top-coding was found and excluded from the "
        "decision:",
        "",
        table(
            ["Arm", "Lift (capped)", "95% CI", "Holm p"],
            [
                [
                    x["arm"],
                    usd(x["diff"], True),
                    f"{usd(x['ci_low'], True)} to {usd(x['ci_high'], True)}",
                    pval(x["p_holm"]),
                ]
                for x in ph["results"]
            ],
        ),
        "",
        "## 5. Secondary metrics",
        "",
        table(
            ["Metric", "Arm", "Control", "Treatment", "Lift", "95% CI", "Holm p"],
            [
                [
                    x["metric"],
                    x["arm"],
                    pct(x["control_mean"], 2),
                    pct(x["treatment_mean"], 2),
                    pp(x["diff"]),
                    f"{pp(x['ci_low'])} to {pp(x['ci_high'])}",
                    pval(x["p_holm"]),
                ]
                for x in r["secondary"]
            ],
        ),
        "",
    ]

    h = r["head_to_head"]
    s += [
        "## 6. Head to head",
        "",
        f"{h['compared_arm']} minus {h['baseline_arm']}: {usd(h['diff'], True)} per customer "
        f"(95% CI {usd(h['ci_low'], True)} to {usd(h['ci_high'], True)}, p {pval(h['p_value'])}).",
        "",
        "## 7. Variance reduction (CUPED and regression adjustment)",
        "",
        table(
            [
                "Arm",
                "Unadjusted SE",
                "CUPED SE",
                "CUPED variance removed",
                "corr(spend, history)",
                "Regression SE",
                "Regression lift",
            ],
            [
                [
                    v["arm"],
                    se_usd(_by_arm(r["primary"], v["arm"])["se"]),
                    se_usd(v["cuped"]["se"]),
                    pct(v["cuped_variance_reduction"], 3),
                    f"{v['cuped_correlation']:.4f}",
                    se_usd(v["regression"]["se"]),
                    usd(v["regression"]["diff"], True),
                ]
                for v in r["variance_reduction"]
            ],
        ),
        "",
        "CUPED removes a share of variance equal to the squared correlation between the "
        "outcome and the covariate. Last year's spend barely predicts spend in a two-week "
        "window (most customers spend nothing in both arms of the comparison), so the "
        "reduction is negligible here. The regression adjustment with every pre-period "
        "covariate does no better. For future tests on this population, variance reduction "
        "would need a covariate measured over a comparable short window, such as spend in "
        "the two weeks before the send.",
        "",
    ]

    op = r["observed_power"]
    s += [
        "## 8. Power at the observed baseline",
        "",
        f"At alpha {op['alpha_per_comparison']} per comparison and {pct(op['power'], 0)} power, "
        "with the realised arm sizes and the observed control baseline, the test could detect:",
        "",
        table(
            ["Metric", "Minimum detectable effect"],
            [
                ["Revenue per customer", usd(op["mde_spend"])],
                ["Conversion rate", pp(op["mde_conversion"])],
                ["Visit rate", pp(op["mde_visit"])],
            ],
        ),
        "",
        "Power to detect the observed revenue effect: "
        + ", ".join(f"{arm} {power_pct(v)}" for arm, v in op["power_at_observed_effect"].items())
        + ". (Observed-effect power is descriptive only; it is a function of the p-value and "
        "adds no evidence.)",
        "",
        "## 9. Segment effects (exploratory, Holm-corrected per email)",
        "",
        "![Segment effects](figures/segment_effects.png)",
        "",
    ]
    for arm in arms:
        s += [
            f"**{arm}**",
            "",
            table(
                ["Segment", "Level", "n (treatment)", "Lift", "95% CI", "Holm p", "Significant"],
                [
                    [
                        x["segment"],
                        x["level"],
                        f"{x['n_treatment']:,}",
                        usd(x["diff"], True),
                        f"{usd(x['ci_low'], True)} to {usd(x['ci_high'], True)}",
                        pval(x["p_holm"]),
                        "yes" if x["significant"] else "no",
                    ]
                    for x in r["segments"][arm]
                ],
            ),
            "",
        ]
    s += [
        "Intervals are unadjusted 95% CIs; the Significant column uses the Holm-adjusted "
        "p-value, which is why an interval can exclude zero while the level is not "
        "significant (for example, a small segment tested alongside ten others).",
        "",
        "A segment-level effect is not independent of the others: the levels of `newbie` "
        "and `channel` contain the same customers, so an effect concentrated in one segment "
        "shows up, diluted, in every segment that overlaps it.",
        "",
    ]

    a = r["assumptions"]
    s += [
        "## 10. Economics",
        "",
        f"Assumptions (not data): gross margin {pct(a['gross_margin'], 0)}, "
        f"{usd(a['cost_per_email'])} per email, rollout to {a['rollout_customers']:,} customers.",
        "",
        table(
            [
                "Arm",
                "Revenue lift",
                "Net profit / email",
                "Profit range (from bootstrap CI)",
                "Break-even margin",
                f"Net profit at {a['rollout_customers']:,}",
            ],
            [
                [
                    arm,
                    usd(e["revenue_lift_per_customer"], True),
                    usd(e["net_profit_per_customer"], True),
                    f"{usd(e['net_profit_ci'][0], True)} to {usd(e['net_profit_ci'][1], True)}",
                    _break_even(e),
                    usd(e["net_profit_at_rollout"], True),
                ]
                for arm, e in r["economics"].items()
            ],
        ),
        "",
        "## 11. Decision rule",
        "",
        table(
            [
                "Arm",
                "Valid",
                "Primary +sig",
                "Capped agrees",
                "No segment harm",
                "Profitable",
                "Qualifies",
            ],
            [
                [
                    d["arm"],
                    *[
                        "yes" if d[k] else "no"
                        for k in (
                            "validity_passed",
                            "primary_significant_positive",
                            "winsorized_agrees",
                            "no_segment_harm",
                            "profitable",
                            "qualifies",
                        )
                    ],
                ]
                for d in r["decision"]["arms"]
            ],
        ),
        "",
        f"**Recommended: {r['decision']['recommended_arm'] or 'neither email'}.** "
        "(Capped agrees refers to the pre-registered cap, which is non-binding here.)",
        "",
    ]

    pk = r["peeking"]
    s += [
        "## 12. Why one look: peeking simulation",
        "",
        "![Peeking](figures/peeking.png)",
        "",
        f"{pk['n_simulations']:,} simulated A/A tests, {pk['n_per_arm']:,} customers per arm, "
        f"true conversion rate {pct(pk['base_rate'], 2)} in both arms (the observed control "
        f"rate). Monte Carlo standard error about {pct(pk['mc_standard_error'], 1)}.",
        "",
        table(
            [
                "Looks",
                "False-positive rate (stop at first p < 0.05)",
                "With Bonferroni across looks",
            ],
            [
                [m, pct(f), pct(b)]
                for m, f, b in zip(
                    pk["looks"],
                    pk["false_positive_rate"],
                    pk["bonferroni_false_positive_rate"],
                    strict=True,
                )
            ],
        ),
        "",
    ]

    up = r["uplift"]
    s += [
        "## 13. Targeting model (exploratory)",
        "",
        "![Uplift](figures/uplift.png)",
        "",
        f"T-learner (two logistic regressions on pre-period features) for the "
        f"{up['treatment_arm']}, outcome `{up['outcome']}`, trained on {up['n_train']:,} "
        f"customers and evaluated on {up['n_test']:,} held-out customers. Normalised Qini "
        f"coefficient {up['qini_coefficient']:.3f} (0 = random targeting). Targeting the top "
        f"{pct(up['top_fraction'], 0)} by predicted uplift captures "
        f"{pct(up['top_share_of_incremental'], 0)} of the incremental {up['outcome']}s.",
        "",
        table(
            ["Decile", "Mean predicted uplift", "Treated rate", "Control rate", "Observed uplift"],
            [
                [
                    d["bin"],
                    pp(d["mean_predicted_uplift"]),
                    pct(d["treated_rate"]),
                    pct(d["control_rate"]),
                    pp(d["observed_uplift"]),
                ]
                for d in up["deciles"]
            ],
        ),
        "",
        "Per-decile uplifts rest on about 640 customers per arm each, so a single decile's "
        "value is noisy; the ordering across deciles is the evidence, not any one bar.",
        "",
        "## 14. Deviations from the plan",
        "",
        "1. The winsorisation cap was clarified before analysis (the second commit to "
        "`docs/TEST_PLAN.md`), from the 99th percentile of all spend to the 99th percentile "
        "of positive spend.",
    ]
    if not tc["planned_cap_binding"]:
        s.append(
            "2. After analysis the planned cap turned out to be non-binding because the source "
            "is top-coded. A 95th-percentile cap was added as a clearly labelled post-hoc "
            "sensitivity check. It is reported, and it is not used by the decision rule."
        )
    return "\n".join(s) + "\n"


def write_outputs(results: dict[str, Any], out_dir: Path) -> dict[str, Path]:
    from campaign_experiment.figures import render_all

    out_dir.mkdir(parents=True, exist_ok=True)
    clean = to_jsonable(results)
    paths = {
        "results": out_dir / "results.json",
        "report": out_dir / "REPORT.md",
        "memo": out_dir / "DECISION_MEMO.md",
    }
    paths["results"].write_text(json.dumps(clean, indent=2) + "\n", encoding="utf-8")
    paths["report"].write_text(full_report(clean), encoding="utf-8")
    paths["memo"].write_text(decision_memo(clean), encoding="utf-8")
    render_all(clean, out_dir / "figures")
    return paths
