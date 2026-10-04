"""Why the analysis looks once: an A/A simulation of peeking.

Both arms share one true rate, so every rejection is a false positive. A
fixed-horizon test rejects about alpha of the time. Checking the running
test at several interim looks and stopping at the first p < alpha gives the
test several chances to cross the line by luck, and the false-positive
rate climbs with the number of looks.

Customers arriving between looks are simulated as a binomial count rather
than one draw per customer: the cumulative conversions are then exactly
distributed as in a row-level simulation, at a cost of n_sims x n_looks
numbers instead of n_sims x n_per_arm.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.stats import norm


@dataclass(frozen=True)
class PeekingResult:
    base_rate: float
    n_per_arm: int
    n_simulations: int
    alpha: float
    looks: list[int]
    false_positive_rate: list[float]
    bonferroni_false_positive_rate: list[float]
    mc_standard_error: float

    def to_dict(self) -> dict:
        return asdict(self)


def _z_stats(conv_c: np.ndarray, conv_t: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Pooled two-proportion z statistic for equal arm sizes, vectorised."""
    pooled = (conv_c + conv_t) / (2 * n)
    se = np.sqrt(pooled * (1 - pooled) * (2 / n))
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (conv_t - conv_c) / (n * se)
    return np.nan_to_num(z, nan=0.0)


def look_schedule(n_looks: int, m: int) -> np.ndarray:
    """Indices of m evenly spaced looks among n_looks, always including the last."""
    if not 1 <= m <= n_looks:
        raise ValueError("m must be between 1 and n_looks")
    return np.unique(np.round(np.linspace(n_looks / m, n_looks, m)).astype(int) - 1)


def simulate_peeking(
    base_rate: float,
    n_per_arm: int,
    n_looks: int,
    n_simulations: int,
    alpha: float,
    seed: int,
    look_counts: list[int] | None = None,
) -> PeekingResult:
    """False-positive rate when a null test is checked at m evenly spaced looks."""
    if not 0 < base_rate < 1:
        raise ValueError("base_rate must be in (0, 1)")
    if n_per_arm < n_looks:
        raise ValueError("need at least one customer per arm between looks")
    rng = np.random.default_rng(seed)
    # Customers per arm at each look, ending exactly at n_per_arm.
    cumulative_n = np.round(np.linspace(n_per_arm / n_looks, n_per_arm, n_looks)).astype(int)
    batch = np.diff(cumulative_n, prepend=0)
    conv_c = rng.binomial(batch, base_rate, size=(n_simulations, n_looks)).cumsum(axis=1)
    conv_t = rng.binomial(batch, base_rate, size=(n_simulations, n_looks)).cumsum(axis=1)
    z = np.abs(_z_stats(conv_c, conv_t, cumulative_n))

    if look_counts is None:
        look_counts = [m for m in (1, 2, 3, 5, 10, 20, n_looks) if m <= n_looks]
    look_counts = sorted(set(look_counts))

    critical = norm.ppf(1 - alpha / 2)
    fpr, bonf = [], []
    for m in look_counts:
        idx = look_schedule(n_looks, m)
        fpr.append(float((z[:, idx] > critical).any(axis=1).mean()))
        # Bonferroni across looks: valid but conservative. Group-sequential
        # boundaries (O'Brien-Fleming, Pocock) spend alpha more efficiently.
        bonf_critical = norm.ppf(1 - alpha / (2 * len(idx)))
        bonf.append(float((z[:, idx] > bonf_critical).any(axis=1).mean()))

    return PeekingResult(
        base_rate=base_rate,
        n_per_arm=n_per_arm,
        n_simulations=n_simulations,
        alpha=alpha,
        looks=look_counts,
        false_positive_rate=fpr,
        bonferroni_false_positive_rate=bonf,
        # Monte Carlo SE of a rate near alpha, to read the table with.
        mc_standard_error=float(np.sqrt(alpha * (1 - alpha) / n_simulations)),
    )
