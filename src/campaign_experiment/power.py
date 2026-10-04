"""Minimum detectable effect and sample size, using the normal approximation.

Planning happens before the outcome is known, so these functions take a
baseline rate or standard deviation as an input assumption. After the test,
the same functions are evaluated at the observed control baseline to state
what the experiment could and could not have detected.
"""

from __future__ import annotations

import math

from scipy.stats import norm


def _z(alpha: float, power: float, two_sided: bool = True) -> tuple[float, float]:
    if not 0 < alpha < 1 or not 0 < power < 1:
        raise ValueError("alpha and power must be in (0, 1)")
    z_alpha = norm.ppf(1 - alpha / 2) if two_sided else norm.ppf(1 - alpha)
    return z_alpha, norm.ppf(power)


def _check_n(*ns: float) -> None:
    if any(n <= 0 for n in ns):
        raise ValueError("sample sizes must be positive")


def mde_mean(sd: float, n_control: int, n_treatment: int, alpha: float, power: float) -> float:
    """Smallest absolute difference in means detectable with the given power."""
    _check_n(n_control, n_treatment)
    if sd <= 0:
        raise ValueError("sd must be positive")
    z_alpha, z_beta = _z(alpha, power)
    return (z_alpha + z_beta) * sd * math.sqrt(1 / n_control + 1 / n_treatment)


def mde_proportion(
    p_base: float, n_control: int, n_treatment: int, alpha: float, power: float
) -> float:
    """Smallest absolute lift in a rate detectable with the given power.

    Uses the baseline variance p(1 - p) for both arms, which is the usual
    planning approximation when the lift is small relative to the baseline.
    """
    if not 0 < p_base < 1:
        raise ValueError("p_base must be in (0, 1)")
    return mde_mean(math.sqrt(p_base * (1 - p_base)), n_control, n_treatment, alpha, power)


def n_per_arm_proportion(p_base: float, lift_abs: float, alpha: float, power: float) -> int:
    """Per-arm sample size for detecting p_base -> p_base + lift_abs (equal arms).

    The textbook two-proportion formula: the null variance uses the pooled
    rate, the alternative variance uses each arm's own rate.
    """
    p2 = p_base + lift_abs
    if not 0 < p_base < 1 or not 0 < p2 < 1 or lift_abs == 0:
        raise ValueError("need 0 < p_base, p_base + lift_abs < 1 and a non-zero lift")
    z_alpha, z_beta = _z(alpha, power)
    p_bar = (p_base + p2) / 2
    numerator = z_alpha * math.sqrt(2 * p_bar * (1 - p_bar)) + z_beta * math.sqrt(
        p_base * (1 - p_base) + p2 * (1 - p2)
    )
    return math.ceil((numerator / lift_abs) ** 2)


def achieved_power_mean(
    diff: float, sd: float, n_control: int, n_treatment: int, alpha: float
) -> float:
    """Power of a two-sided test to detect a true difference `diff`."""
    _check_n(n_control, n_treatment)
    se = sd * math.sqrt(1 / n_control + 1 / n_treatment)
    z_alpha = norm.ppf(1 - alpha / 2)
    shift = abs(diff) / se
    return float(norm.cdf(shift - z_alpha) + norm.cdf(-shift - z_alpha))
