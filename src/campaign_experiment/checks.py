"""Validity checks that must pass before any effect estimate is read."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.stats import chisquare

NUMERIC_COVARIATES = ["recency", "history", "mens", "womens", "newbie"]
CATEGORICAL_COVARIATES = ["zip_code", "channel"]


@dataclass(frozen=True)
class SRMResult:
    observed: dict[str, int]
    expected: dict[str, float]
    chi2: float
    p_value: float
    alpha: float

    @property
    def passed(self) -> bool:
        return self.p_value >= self.alpha

    def to_dict(self) -> dict:
        return {**asdict(self), "passed": self.passed}


def srm_check(counts: dict[str, int], expected_shares: list[float], alpha: float) -> SRMResult:
    """Chi-square goodness of fit of arm counts against the planned allocation.

    A mismatch means assignment or logging is broken, and then no effect
    estimate can be trusted, however significant it looks.
    """
    arms = list(counts)
    if len(arms) != len(expected_shares):
        raise ValueError("need one expected share per arm")
    observed = np.array([counts[a] for a in arms], dtype=float)
    if (observed <= 0).any():
        raise ValueError("every arm needs at least one unit")
    expected = observed.sum() * np.asarray(expected_shares, dtype=float)
    stat, p = chisquare(observed, expected)
    return SRMResult(
        observed={a: int(o) for a, o in zip(arms, observed, strict=True)},
        expected={a: float(e) for a, e in zip(arms, expected, strict=True)},
        chi2=float(stat),
        p_value=float(p),
        alpha=alpha,
    )


def _covariate_matrix(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df[NUMERIC_COVARIATES].astype(float)
    dummies = pd.get_dummies(df[CATEGORICAL_COVARIATES], dtype=float)
    return pd.concat([numeric, dummies], axis=1)


def covariate_balance(
    df: pd.DataFrame, control: str, treatments: list[str], threshold: float
) -> pd.DataFrame:
    """Standardised mean difference of every pre-period covariate, per treatment.

    SMD = (mean_t - mean_c) / sqrt((var_t + var_c) / 2). Unlike a p-value it
    does not shrink toward "significant" as n grows, so it measures whether
    the arms are comparable rather than whether the sample is large.
    """
    X = _covariate_matrix(df)
    arm = df["segment"]
    c = X[arm == control]
    rows = []
    for t in treatments:
        tr = X[arm == t]
        pooled_sd = np.sqrt((tr.var() + c.var()) / 2)
        smd = (tr.mean() - c.mean()) / pooled_sd.replace(0, np.nan)
        for cov, value in smd.items():
            rows.append(
                {
                    "treatment": t,
                    "covariate": cov,
                    "mean_control": float(c[cov].mean()),
                    "mean_treatment": float(tr[cov].mean()),
                    "smd": float(value),
                }
            )
    out = pd.DataFrame(rows)
    out["balanced"] = out["smd"].abs() <= threshold
    return out
