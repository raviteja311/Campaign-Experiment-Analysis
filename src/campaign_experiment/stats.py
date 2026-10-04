"""Hypothesis tests, confidence intervals and multiplicity correction.

The parametric tests take sufficient statistics (mean, variance, count),
which is exactly what the SQL layer returns, so the same function serves
arm-level and segment-level comparisons. The bootstrap needs row-level data
and is used as an approximation-free check on the primary metric.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
from scipy import stats as sps


@dataclass(frozen=True)
class Estimate:
    """Treatment minus control for one metric, with its uncertainty."""

    metric: str
    method: str
    n_control: int
    n_treatment: int
    control_mean: float
    treatment_mean: float
    diff: float
    se: float
    ci_low: float
    ci_high: float
    statistic: float
    p_value: float

    @property
    def relative_lift(self) -> float:
        return self.diff / self.control_mean if self.control_mean else math.nan

    def to_dict(self) -> dict:
        return {**asdict(self), "relative_lift": self.relative_lift}


def welch_test(
    mean_c: float,
    var_c: float,
    n_c: int,
    mean_t: float,
    var_t: float,
    n_t: int,
    alpha: float,
    metric: str = "",
) -> Estimate:
    """Welch's unequal-variance t-test and CI from summary statistics."""
    if n_c < 2 or n_t < 2:
        raise ValueError("Welch's test needs at least two units per arm")
    vc, vt = var_c / n_c, var_t / n_t
    se = math.sqrt(vc + vt)
    diff = mean_t - mean_c
    if se == 0:
        # Both arms constant: no sampling variability to test against.
        p = 1.0 if diff == 0 else 0.0
        return Estimate(
            metric, "welch", n_c, n_t, mean_c, mean_t, diff, 0.0, diff, diff, math.nan, p
        )
    # Welch-Satterthwaite degrees of freedom.
    dof = (vc + vt) ** 2 / (vc**2 / (n_c - 1) + vt**2 / (n_t - 1))
    t = diff / se
    p = 2 * sps.t.sf(abs(t), dof)
    half = sps.t.ppf(1 - alpha / 2, dof) * se
    return Estimate(
        metric, "welch", n_c, n_t, mean_c, mean_t, diff, se, diff - half, diff + half, t, p
    )


def two_proportion_ztest(
    x_c: int, n_c: int, x_t: int, n_t: int, alpha: float, metric: str = ""
) -> Estimate:
    """Two-sided z-test for a difference in rates.

    The test statistic uses the pooled rate (the null says the rates are
    equal); the confidence interval uses each arm's own rate (it describes
    the difference, so it must not assume it is zero).
    """
    if n_c <= 0 or n_t <= 0 or not (0 <= x_c <= n_c and 0 <= x_t <= n_t):
        raise ValueError("need 0 <= successes <= n and n > 0 in both arms")
    p_c, p_t = x_c / n_c, x_t / n_t
    diff = p_t - p_c
    pooled = (x_c + x_t) / (n_c + n_t)
    se_null = math.sqrt(pooled * (1 - pooled) * (1 / n_c + 1 / n_t))
    se = math.sqrt(p_c * (1 - p_c) / n_c + p_t * (1 - p_t) / n_t)
    z = diff / se_null if se_null > 0 else 0.0
    p = 2 * sps.norm.sf(abs(z))
    half = sps.norm.ppf(1 - alpha / 2) * se
    return Estimate(metric, "z", n_c, n_t, p_c, p_t, diff, se, diff - half, diff + half, z, p)


def bootstrap_diff_ci(
    y_control: np.ndarray,
    y_treatment: np.ndarray,
    n_resamples: int,
    confidence: float,
    seed: int,
    chunk: int = 100,
) -> tuple[float, float, np.ndarray]:
    """Percentile bootstrap CI for mean(treatment) - mean(control).

    Each arm is resampled separately, because randomisation fixed the arm
    sizes. Resamples are drawn in chunks so memory stays bounded at roughly
    chunk x n integers regardless of n_resamples.
    """
    y_c = np.asarray(y_control, dtype=float)
    y_t = np.asarray(y_treatment, dtype=float)
    if len(y_c) < 2 or len(y_t) < 2:
        raise ValueError("bootstrap needs at least two units per arm")
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_resamples)
    done = 0
    while done < n_resamples:
        k = min(chunk, n_resamples - done)
        mean_c = y_c[rng.integers(0, len(y_c), size=(k, len(y_c)))].mean(axis=1)
        mean_t = y_t[rng.integers(0, len(y_t), size=(k, len(y_t)))].mean(axis=1)
        diffs[done : done + k] = mean_t - mean_c
        done += k
    tail = (1 - confidence) / 2 * 100
    low, high = np.percentile(diffs, [tail, 100 - tail])
    return float(low), float(high), diffs


def holm(p_values: list[float]) -> list[float]:
    """Holm step-down adjusted p-values (family-wise error control).

    Uniformly more powerful than Bonferroni and valid under any dependence
    between the tests, which matters because segment tests share customers'
    control arm.
    """
    p = np.asarray(p_values, dtype=float)
    if p.size == 0:
        return []
    if np.isnan(p).any() or (p < 0).any() or (p > 1).any():
        raise ValueError("p-values must be in [0, 1]")
    m = p.size
    order = np.argsort(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adjusted[idx] = min(running, 1.0)
    return adjusted.tolist()


def winsorize(values: np.ndarray, cap: float) -> np.ndarray:
    """Cap values at `cap`. The cap is computed once on the pooled sample by
    the caller, so treatment and control are capped at the same dollar value."""
    return np.minimum(np.asarray(values, dtype=float), cap)
