"""Use pre-period data to estimate the same effect with less noise.

Randomisation makes pre-period covariates independent of treatment, so
subtracting their predictable part from the outcome leaves the expected
difference unchanged and removes variance. Two estimators:

- CUPED: one covariate, theta = cov(Y, X) / var(X).
- Regression adjustment (Lin 2013): OLS of Y on treatment, centred
  covariates and their interactions, with HC2 robust standard errors.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm

from campaign_experiment.stats import Estimate, welch_test


@dataclass(frozen=True)
class CupedResult:
    estimate: Estimate
    theta: float
    variance_reduction: float
    correlation: float


def cuped_theta(y: np.ndarray, x: np.ndarray) -> float:
    var_x = np.var(x, ddof=1)
    if var_x == 0:
        raise ValueError("CUPED covariate has no variance")
    return float(np.cov(y, x, ddof=1)[0, 1] / var_x)


def cuped_effect(
    df: pd.DataFrame,
    metric: str,
    covariate: str,
    control: str,
    treatment: str,
    alpha: float,
) -> CupedResult:
    """CUPED-adjusted treatment effect.

    theta and the covariate mean are estimated on the pooled sample of the
    two arms being compared, so both arms receive the identical adjustment.
    """
    sub = df[df["segment"].isin([control, treatment])]
    y = sub[metric].to_numpy(dtype=float)
    x = sub[covariate].to_numpy(dtype=float)
    theta = cuped_theta(y, x)
    adjusted = y - theta * (x - x.mean())
    is_t = (sub["segment"] == treatment).to_numpy()
    yc, yt = adjusted[~is_t], adjusted[is_t]
    est = welch_test(
        yc.mean(), yc.var(ddof=1), len(yc), yt.mean(), yt.var(ddof=1), len(yt), alpha, metric
    )
    # Report the arms' raw means; only the difference and its SE are adjusted.
    raw_c = y[~is_t].mean()
    est = Estimate(
        metric,
        "cuped",
        est.n_control,
        est.n_treatment,
        raw_c,
        raw_c + est.diff,
        est.diff,
        est.se,
        est.ci_low,
        est.ci_high,
        est.statistic,
        est.p_value,
    )
    reduction = 1 - adjusted.var(ddof=1) / y.var(ddof=1)
    return CupedResult(est, theta, float(reduction), float(np.corrcoef(y, x)[0, 1]))


def regression_adjusted_effect(
    df: pd.DataFrame,
    metric: str,
    control: str,
    treatment: str,
    numeric: list[str],
    categorical: list[str],
    alpha: float,
) -> Estimate:
    """Lin (2013) estimator: interacted OLS with centred covariates, HC2 SEs.

    Centring makes the treatment coefficient the average effect; interacting
    lets the covariate slopes differ by arm, which keeps the estimator
    consistent even if the linear model is wrong.
    """
    sub = df[df["segment"].isin([control, treatment])]
    covs = pd.concat(
        [
            sub[numeric].astype(float),
            pd.get_dummies(sub[categorical], drop_first=True, dtype=float),
        ],
        axis=1,
    )
    covs = covs - covs.mean()
    treat = (sub["segment"] == treatment).astype(float).rename("treatment")
    interactions = covs.mul(treat, axis=0).add_suffix(":treatment")
    X = sm.add_constant(pd.concat([treat, covs, interactions], axis=1))
    y = sub[metric].astype(float)
    fit = sm.OLS(y, X).fit(cov_type="HC2")
    diff = float(fit.params["treatment"])
    se = float(fit.bse["treatment"])
    low, high = fit.conf_int(alpha=alpha).loc["treatment"]
    mean_c = float(y[treat == 0].mean())
    return Estimate(
        metric,
        "regression_adjusted",
        int((treat == 0).sum()),
        int((treat == 1).sum()),
        mean_c,
        mean_c + diff,
        diff,
        se,
        float(low),
        float(high),
        float(fit.tvalues["treatment"]),
        float(fit.pvalues["treatment"]),
    )
