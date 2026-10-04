import math

import pytest
from statsmodels.stats.power import NormalIndPower

from campaign_experiment.power import (
    achieved_power_mean,
    mde_mean,
    mde_proportion,
    n_per_arm_proportion,
)


def test_mde_mean_gives_requested_power_per_statsmodels():
    sd, n = 15.0, 21_000
    mde = mde_mean(sd, n, n, alpha=0.05, power=0.8)
    power = NormalIndPower().power(effect_size=mde / sd, nobs1=n, alpha=0.05, ratio=1.0)
    assert power == pytest.approx(0.8, abs=1e-3)


def test_mde_shrinks_with_root_n():
    small = mde_mean(10, 1000, 1000, 0.05, 0.8)
    large = mde_mean(10, 4000, 4000, 0.05, 0.8)
    assert small / large == pytest.approx(2.0)


def test_mde_proportion_uses_bernoulli_sd():
    p = 0.01
    assert mde_proportion(p, 5000, 5000, 0.05, 0.8) == pytest.approx(
        mde_mean(math.sqrt(p * (1 - p)), 5000, 5000, 0.05, 0.8)
    )


def test_n_per_arm_roundtrip_with_mde():
    p, lift = 0.10, 0.01
    n = n_per_arm_proportion(p, lift, 0.05, 0.8)
    # Known textbook value for 10% -> 11% at 80% power, alpha 0.05: ~14,750 per arm.
    assert 14_500 < n < 15_000
    # At that n, the MDE is close to the planned lift.
    assert mde_proportion(p, n, n, 0.05, 0.8) == pytest.approx(lift, rel=0.05)


def test_achieved_power_at_mde_is_requested_power():
    mde = mde_mean(12, 21_000, 21_300, 0.05, 0.9)
    assert achieved_power_mean(mde, 12, 21_000, 21_300, 0.05) == pytest.approx(0.9, abs=1e-3)


def test_achieved_power_at_zero_effect_is_alpha():
    assert achieved_power_mean(0.0, 12, 1000, 1000, 0.05) == pytest.approx(0.05)


@pytest.mark.parametrize(
    "call",
    [
        lambda: mde_mean(0, 100, 100, 0.05, 0.8),
        lambda: mde_mean(1, 0, 100, 0.05, 0.8),
        lambda: mde_proportion(1.2, 100, 100, 0.05, 0.8),
        lambda: n_per_arm_proportion(0.5, 0.0, 0.05, 0.8),
        lambda: mde_mean(1, 100, 100, 1.5, 0.8),
    ],
)
def test_invalid_inputs_rejected(call):
    with pytest.raises(ValueError):
        call()
