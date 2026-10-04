import numpy as np
import pytest
from scipy import stats as sps
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import confint_proportions_2indep, proportions_ztest

from campaign_experiment.stats import (
    bootstrap_diff_ci,
    holm,
    two_proportion_ztest,
    welch_test,
    winsorize,
)


def _welch_rows(yc, yt, alpha=0.05):
    return welch_test(yc.mean(), yc.var(ddof=1), len(yc), yt.mean(), yt.var(ddof=1), len(yt), alpha)


def test_welch_matches_scipy():
    rng = np.random.default_rng(1)
    yc = rng.exponential(10, 800)
    yt = rng.exponential(11, 1200)
    ours = _welch_rows(yc, yt)
    ref = sps.ttest_ind(yt, yc, equal_var=False)
    assert ours.statistic == pytest.approx(ref.statistic)
    assert ours.p_value == pytest.approx(ref.pvalue)
    low, high = ref.confidence_interval(0.95)
    assert (ours.ci_low, ours.ci_high) == pytest.approx((low, high))


def test_welch_constant_arms():
    est = welch_test(1.0, 0.0, 10, 1.0, 0.0, 10, 0.05)
    assert est.p_value == 1.0 and est.se == 0.0


def test_welch_needs_two_units():
    with pytest.raises(ValueError):
        welch_test(1.0, 1.0, 1, 1.0, 1.0, 10, 0.05)


def test_ztest_matches_statsmodels():
    x_c, n_c, x_t, n_t = 1500, 21306, 1800, 21387
    ours = two_proportion_ztest(x_c, n_c, x_t, n_t, 0.05)
    z, p = proportions_ztest([x_t, x_c], [n_t, n_c])
    assert ours.statistic == pytest.approx(z)
    assert ours.p_value == pytest.approx(p)
    low, high = confint_proportions_2indep(x_t, n_t, x_c, n_c, method="wald")
    assert (ours.ci_low, ours.ci_high) == pytest.approx((low, high))
    assert ours.relative_lift == pytest.approx((x_t / n_t) / (x_c / n_c) - 1)


@pytest.mark.parametrize("args", [(5, 0, 5, 10), (11, 10, 5, 10), (-1, 10, 5, 10)])
def test_ztest_rejects_bad_counts(args):
    with pytest.raises(ValueError):
        two_proportion_ztest(*args, alpha=0.05)


def test_ztest_recovers_known_lift():
    rng = np.random.default_rng(2)
    n = 20_000
    c = rng.random(n) < 0.15
    t = rng.random(n) < 0.17
    est = two_proportion_ztest(int(c.sum()), n, int(t.sum()), n, 0.05)
    assert est.p_value < 0.001
    assert est.ci_low < 0.02 < est.ci_high


def test_false_positive_rate_is_nominal_under_null():
    """A/A: with no true effect, about 5% of tests reject at alpha 0.05."""
    rng = np.random.default_rng(3)
    rejections = 0
    sims = 2000
    for _ in range(sims):
        yc = rng.exponential(10, 200)
        yt = rng.exponential(10, 200)
        rejections += _welch_rows(yc, yt).p_value < 0.05
    rate = rejections / sims
    assert 0.035 < rate < 0.065


def test_bootstrap_agrees_with_welch_on_large_samples():
    rng = np.random.default_rng(4)
    yc = rng.exponential(10, 5000)
    yt = rng.exponential(10.5, 5000)
    low, high, diffs = bootstrap_diff_ci(yc, yt, 2000, 0.95, seed=1)
    welch = _welch_rows(yc, yt)
    assert len(diffs) == 2000
    assert low == pytest.approx(welch.ci_low, abs=0.08)
    assert high == pytest.approx(welch.ci_high, abs=0.08)


def test_bootstrap_is_deterministic_and_chunk_independent():
    rng = np.random.default_rng(5)
    yc, yt = rng.normal(0, 1, 300), rng.normal(0, 1, 300)
    a = bootstrap_diff_ci(yc, yt, 250, 0.9, seed=9, chunk=100)[:2]
    b = bootstrap_diff_ci(yc, yt, 250, 0.9, seed=9, chunk=100)[:2]
    assert a == b


def test_bootstrap_needs_data():
    with pytest.raises(ValueError):
        bootstrap_diff_ci(np.array([1.0]), np.array([1.0, 2.0]), 100, 0.95, seed=0)


def test_holm_matches_statsmodels():
    p = [0.01, 0.04, 0.03, 0.005, 0.2, 0.049]
    expected = multipletests(p, method="holm")[1]
    assert holm(p) == pytest.approx(list(expected))


def test_holm_edge_cases():
    assert holm([]) == []
    assert holm([0.9, 0.8]) == [1.0, 1.0]
    assert holm([0.04, 0.3]) == pytest.approx([0.08, 0.3])
    with pytest.raises(ValueError):
        holm([0.1, 1.2])


def test_winsorize_caps_both_arms_at_same_value():
    assert winsorize(np.array([1.0, 50.0, 500.0]), 100.0).tolist() == [1.0, 50.0, 100.0]
