"""Variance reduction, segment effects, economics and the decision rule."""

import math

import numpy as np
import pandas as pd
import pytest
from conftest import ARMS, make_experiment

from campaign_experiment.checks import CATEGORICAL_COVARIATES, NUMERIC_COVARIATES
from campaign_experiment.data import add_features
from campaign_experiment.decision import ArmDecision, decide
from campaign_experiment.economics import campaign_economics
from campaign_experiment.metrics_sql import connect, segment_metrics
from campaign_experiment.segments import segment_effects
from campaign_experiment.stats import welch_test
from campaign_experiment.variance_reduction import (
    cuped_effect,
    cuped_theta,
    regression_adjusted_effect,
)

CONTROL, MENS, WOMENS = ARMS


@pytest.fixture
def correlated():
    """Spend strongly driven by pre-period history, plus a $40 lift on Womens converters."""
    raw = make_experiment(
        n_per_arm=6000,
        visit_rate={CONTROL: 0.95, MENS: 0.95, WOMENS: 0.95},
        conversion_given_visit=0.95,
        spend_lift={CONTROL: 0.0, MENS: 0.0, WOMENS: 40.0},
        history_effect=1.0,
        seed=11,
    )
    return add_features(raw, ARMS)


def _unadjusted(df, treatment):
    c = df.loc[df["segment"] == CONTROL, "spend"]
    t = df.loc[df["segment"] == treatment, "spend"]
    return welch_test(c.mean(), c.var(), len(c), t.mean(), t.var(), len(t), 0.05)


def test_cuped_reduces_variance_without_moving_the_estimate(correlated):
    raw = _unadjusted(correlated, WOMENS)
    cuped = cuped_effect(correlated, "spend", "history", CONTROL, WOMENS, 0.05)
    true_lift = 0.95 * 0.95 * 40  # visit x conversion x lift on converters
    assert cuped.variance_reduction > 0.2
    # With one covariate, CUPED removes a share of variance equal to rho squared.
    assert cuped.variance_reduction == pytest.approx(cuped.correlation**2, abs=0.02)
    assert cuped.estimate.se < raw.se
    assert cuped.estimate.diff == pytest.approx(true_lift, abs=3 * cuped.estimate.se)
    assert cuped.estimate.method == "cuped"
    assert cuped.correlation > 0.4


def test_cuped_is_unbiased_under_the_null():
    """Across A/A replications the CUPED estimate averages to zero."""
    diffs = []
    for seed in range(40):
        df = add_features(make_experiment(n_per_arm=1500, history_effect=0.6, seed=seed), ARMS)
        diffs.append(cuped_effect(df, "spend", "history", CONTROL, MENS, 0.05).estimate.diff)
    diffs = np.array(diffs)
    assert abs(diffs.mean()) < 3 * diffs.std(ddof=1) / math.sqrt(len(diffs))


def test_cuped_theta_requires_variance():
    with pytest.raises(ValueError):
        cuped_theta(np.array([1.0, 2.0, 3.0]), np.array([5.0, 5.0, 5.0]))


def test_regression_adjustment_agrees_with_cuped(correlated):
    reg = regression_adjusted_effect(
        correlated,
        "spend",
        CONTROL,
        WOMENS,
        NUMERIC_COVARIATES,
        CATEGORICAL_COVARIATES,
        0.05,
    )
    cuped = cuped_effect(correlated, "spend", "history", CONTROL, WOMENS, 0.05).estimate
    assert reg.method == "regression_adjusted"
    assert reg.diff == pytest.approx(cuped.diff, abs=2 * cuped.se)
    assert reg.se < _unadjusted(correlated, WOMENS).se
    assert reg.ci_low < reg.diff < reg.ci_high


def test_segment_effects_localise_harm_to_its_level():
    raw = make_experiment(
        n_per_arm=6000,
        visit_rate={CONTROL: 0.4, MENS: 0.4, WOMENS: 0.4},
        conversion_given_visit=0.6,
        seed=21,
    )
    # Mens email wipes out purchases among Web customers.
    hurt = (raw["segment"] == MENS) & (raw["channel"] == "Web")
    raw.loc[hurt, ["conversion", "spend"]] = [0, 0.0]
    df = add_features(raw, ARMS)
    seg = segment_metrics(connect(df), ["channel", "newbie"])
    effects = segment_effects(seg, CONTROL, MENS, 0.05)
    harm = effects.set_index(["segment", "level"])["harm"]
    # Web is harmed and the other channels are not. Both newbie levels may be
    # flagged too, correctly: each contains Web customers, so it inherits part
    # of the loss. A segment-level harm can be a symptom of another segment.
    assert harm[("channel", "Web")]
    assert not harm[("channel", "Phone")] and not harm[("channel", "Multichannel")]
    assert (effects["p_holm"] >= effects["p_value"]).all()
    assert len(effects) == 3 + 2


def test_segment_effects_skip_levels_missing_an_arm():
    seg = pd.DataFrame(
        {
            "segment": ["channel"] * 2,
            "level": ["Web"] * 2,
            "arm": [CONTROL, WOMENS],
            "n": [50, 50],
            "revenue_per_customer": [1.0, 2.0],
            "revenue_var": [1.0, 1.0],
        }
    )
    assert segment_effects(seg, CONTROL, MENS, 0.05).empty


def test_economics_profit_and_break_even():
    econ = campaign_economics(1.0, (0.5, 1.5), 0.4, 0.05, 100_000)
    assert econ.net_profit_per_customer == pytest.approx(0.35)
    assert econ.net_profit_ci == pytest.approx((0.15, 0.55))
    assert econ.break_even_margin == pytest.approx(0.05)
    assert econ.return_per_dollar == pytest.approx(8.0)
    assert econ.net_profit_at_rollout == pytest.approx(35_000)


def test_economics_without_a_lift_has_no_break_even():
    econ = campaign_economics(-0.2, (-0.5, 0.1), 0.4, 0.05, 1000)
    assert econ.break_even_margin is None
    assert econ.net_profit_per_customer < 0


def test_economics_tiny_lift_never_breaks_even():
    assert campaign_economics(0.01, (0.0, 0.02), 0.4, 0.05, 1).break_even_margin == math.inf


def test_economics_rejects_bad_assumptions():
    with pytest.raises(ValueError):
        campaign_economics(1.0, (0.5, 1.5), 1.5, 0.05, 10)


def _arm(name, **failures):
    base = dict(
        validity_passed=True,
        primary_significant_positive=True,
        winsorized_agrees=True,
        no_segment_harm=True,
        profitable=True,
    )
    return ArmDecision(name, **{**base, **failures})


def test_decide_picks_most_profitable_qualifier():
    arms = [_arm(MENS), _arm(WOMENS)]
    assert decide(arms, {MENS: 0.3, WOMENS: 0.5}) == WOMENS


def test_decide_ignores_non_qualifying_arms():
    arms = [_arm(MENS), _arm(WOMENS, no_segment_harm=False)]
    assert decide(arms, {MENS: 0.3, WOMENS: 0.5}) == MENS
    assert arms[1].failed_criteria == ["no_segment_harm"]


def test_decide_returns_none_when_nothing_qualifies():
    assert decide([_arm(MENS, profitable=False)], {MENS: -0.1}) is None
