"""Synthetic experiments with known effects.

Every statistical test in this suite runs on data whose true effect is set
here, so a method is checked against the answer it should recover rather
than against its own output. The generated frames satisfy every integrity
rule in data.validate, so they exercise the real pipeline end to end.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from campaign_experiment.config import load_config
from campaign_experiment.data import HISTORY_BANDS, add_features

ARMS = ["No E-Mail", "Mens E-Mail", "Womens E-Mail"]


def _history_segment(history: np.ndarray) -> list[str]:
    labels = []
    for value in history:
        for label, (lo, hi) in HISTORY_BANDS.items():
            if lo <= value < hi:
                labels.append(label)
                break
    return labels


def make_experiment(
    n_per_arm: int = 3000,
    visit_rate: dict[str, float] | None = None,
    conversion_given_visit: float = 0.08,
    spend_lift: dict[str, float] | None = None,
    history_effect: float = 0.0,
    seed: int = 0,
    allocation: dict[str, int] | None = None,
) -> pd.DataFrame:
    """A raw (unvalidated) Hillstrom-shaped frame.

    visit_rate: per-arm visit probability.
    spend_lift: per-arm additive dollars on converters' order value.
    history_effect: how strongly pre-period history drives spend, which is
        what gives CUPED something to remove.
    allocation: per-arm row counts, to simulate a sample-ratio mismatch.
    """
    rng = np.random.default_rng(seed)
    visit_rate = visit_rate or {arm: 0.15 for arm in ARMS}
    spend_lift = spend_lift or {arm: 0.0 for arm in ARMS}
    allocation = allocation or {arm: n_per_arm for arm in ARMS}

    frames = []
    for arm in ARMS:
        n = allocation[arm]
        history = np.round(rng.gamma(shape=1.5, scale=160, size=n) + 29.99, 2)
        line = rng.choice(["mens_only", "womens_only", "both"], size=n, p=[0.45, 0.45, 0.10])
        visit = (rng.random(n) < visit_rate[arm]).astype(int)
        conversion = visit * (rng.random(n) < conversion_given_visit).astype(int)
        order_value = 60 + spend_lift[arm] + history_effect * history + rng.gamma(2, 30, size=n)
        spend = np.where(conversion == 1, np.round(order_value, 2), 0.0)
        frames.append(
            pd.DataFrame(
                {
                    "recency": rng.integers(1, 13, size=n),
                    "history_segment": _history_segment(history),
                    "history": history,
                    "mens": (line != "womens_only").astype(int),
                    "womens": (line != "mens_only").astype(int),
                    "zip_code": rng.choice(["Rural", "Surburban", "Urban"], size=n),
                    "newbie": rng.integers(0, 2, size=n),
                    "channel": rng.choice(["Multichannel", "Phone", "Web"], size=n),
                    "segment": arm,
                    "visit": visit,
                    "conversion": conversion,
                    "spend": spend,
                }
            )
        )
    return (
        pd.concat(frames, ignore_index=True)
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )


@pytest.fixture
def cfg():
    return load_config()


@pytest.fixture
def raw_frame():
    return make_experiment()


@pytest.fixture
def frame(raw_frame):
    return add_features(raw_frame, ARMS)
