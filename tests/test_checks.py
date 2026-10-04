import pytest
from conftest import ARMS, make_experiment

from campaign_experiment.checks import covariate_balance, srm_check
from campaign_experiment.data import add_features

THIRDS = [1 / 3, 1 / 3, 1 / 3]


def test_srm_passes_on_planned_split():
    result = srm_check({"a": 21306, "b": 21307, "c": 21387}, THIRDS, alpha=0.001)
    assert result.passed
    assert result.to_dict()["passed"] is True


def test_srm_fails_on_mismatch():
    # A 10% shortfall in one arm at this size is far outside chance.
    result = srm_check({"a": 3000, "b": 3000, "c": 2700}, THIRDS, alpha=0.001)
    assert not result.passed
    assert result.p_value < 1e-4


def test_srm_respects_unequal_plan():
    result = srm_check({"a": 5000, "b": 2500, "c": 2500}, [0.5, 0.25, 0.25], alpha=0.001)
    assert result.passed
    assert result.expected["a"] == pytest.approx(5000)


@pytest.mark.parametrize(
    "counts,shares", [({"a": 1, "b": 1}, THIRDS), ({"a": 0, "b": 5, "c": 5}, THIRDS)]
)
def test_srm_rejects_bad_input(counts, shares):
    with pytest.raises(ValueError):
        srm_check(counts, shares, alpha=0.001)


def test_randomised_arms_are_balanced(frame):
    bal = covariate_balance(frame, ARMS[0], ARMS[1:], threshold=0.1)
    assert bal["balanced"].all()
    assert set(bal["treatment"]) == set(ARMS[1:])
    # Categoricals are one-hot encoded, one row per level.
    assert {"channel_Web", "zip_code_Urban"} <= set(bal["covariate"])


def test_imbalance_is_flagged():
    raw = make_experiment(n_per_arm=4000, seed=3)
    womens = raw["segment"] == "Womens E-Mail"
    raw.loc[womens, "recency"] = (raw.loc[womens, "recency"] + 3).clip(upper=12)
    bal = covariate_balance(add_features(raw, ARMS), ARMS[0], ARMS[1:], threshold=0.1)
    flagged = bal[~bal["balanced"]]
    assert list(flagged["covariate"]) == ["recency"]
    assert list(flagged["treatment"]) == ["Womens E-Mail"]
