import numpy as np
import pytest
from conftest import ARMS, make_experiment

from campaign_experiment.data import add_features
from campaign_experiment.peeking import look_schedule, simulate_peeking
from campaign_experiment.uplift import decile_table, fit_t_learner, qini_coefficient, qini_curve

CONTROL, MENS, WOMENS = ARMS


# --- peeking --------------------------------------------------------------


@pytest.fixture(scope="module")
def peeking():
    return simulate_peeking(
        base_rate=0.1,
        n_per_arm=5000,
        n_looks=20,
        n_simulations=4000,
        alpha=0.05,
        seed=1,
        look_counts=[1, 5, 20],
    )


def test_single_look_holds_alpha(peeking):
    assert peeking.false_positive_rate[0] == pytest.approx(0.05, abs=4 * peeking.mc_standard_error)


def test_peeking_inflates_false_positives(peeking):
    fpr = peeking.false_positive_rate
    assert fpr[0] < fpr[1] < fpr[2]
    # Twenty looks at least doubles the nominal 5%.
    assert fpr[2] > 0.10


def test_bonferroni_keeps_peeking_at_or_below_alpha(peeking):
    limit = 0.05 + 4 * peeking.mc_standard_error
    assert all(rate <= limit for rate in peeking.bonferroni_false_positive_rate)


def test_look_schedule_ends_at_final_look():
    assert look_schedule(20, 1).tolist() == [19]
    assert look_schedule(20, 4).tolist() == [4, 9, 14, 19]
    with pytest.raises(ValueError):
        look_schedule(20, 21)


def test_peeking_rejects_bad_inputs():
    with pytest.raises(ValueError):
        simulate_peeking(1.5, 100, 5, 10, 0.05, 0)
    with pytest.raises(ValueError):
        simulate_peeking(0.1, 3, 5, 10, 0.05, 0)


# --- uplift ---------------------------------------------------------------


def test_qini_perfect_and_random_rankings():
    rng = np.random.default_rng(0)
    n = 4000
    responder = rng.random(n) < 0.3
    treated = rng.integers(0, 2, n)
    # Only responders react to treatment, and only when treated.
    y = (responder & (treated == 1)).astype(int)
    perfect = qini_coefficient(qini_curve(responder.astype(float), treated, y))
    random = qini_coefficient(qini_curve(rng.random(n), treated, y))
    assert perfect > 0.05
    assert abs(random) < perfect / 5


def test_decile_table_shape_and_ordering():
    rng = np.random.default_rng(1)
    n = 2000
    score = rng.random(n)
    treated = rng.integers(0, 2, n)
    y = ((score > 0.5) & (treated == 1)).astype(int)
    table = decile_table(score, treated, y, n_bins=10)
    assert len(table) == 10
    assert (table["n_treated"] + table["n_control"]).sum() == n
    # Bin 1 holds the highest scores, where the effect lives.
    assert table["observed_uplift"].iloc[0] > table["observed_uplift"].iloc[-1]


def test_t_learner_finds_the_responsive_segment():
    raw = make_experiment(n_per_arm=8000, seed=5)
    # Womens email raises visits only among newbies.
    rng = np.random.default_rng(5)
    boost = (raw["segment"] == WOMENS) & (raw["newbie"] == 1) & (rng.random(len(raw)) < 0.25)
    raw.loc[boost, "visit"] = 1
    df = add_features(raw, ARMS)
    result = fit_t_learner(df, CONTROL, WOMENS, "visit", 0.3, seed=0, n_bins=5)
    deciles = result.deciles
    assert result.qini_coefficient > 0
    assert deciles["observed_uplift"].iloc[0] > deciles["observed_uplift"].iloc[-1]
    assert result.n_train + result.n_test == 16_000
    assert 0 < result.top_share_of_incremental
