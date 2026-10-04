"""Exploratory targeting: who responds to the email? (T-learner)

Two classifiers are fitted on the training split, one on treated customers
and one on control customers. A customer's predicted uplift is the
difference of the two predicted probabilities. Uplift is never observed for
an individual, so the model is judged on the held-out split by whether the
customers it ranks highest show a larger *observed* treatment-minus-control
difference than the customers it ranks lowest.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from campaign_experiment.checks import CATEGORICAL_COVARIATES, NUMERIC_COVARIATES


@dataclass(frozen=True)
class UpliftResult:
    treatment_arm: str
    outcome: str
    n_train: int
    n_test: int
    deciles: pd.DataFrame
    qini_curve: pd.DataFrame
    qini_coefficient: float
    top_share_of_incremental: float
    top_fraction: float


def features(df: pd.DataFrame) -> pd.DataFrame:
    """Pre-period covariates only: nothing measured after the email was sent."""
    return pd.concat(
        [
            df[NUMERIC_COVARIATES].astype(float),
            pd.get_dummies(df[CATEGORICAL_COVARIATES], dtype=float),
        ],
        axis=1,
    )


def _model(seed: int):
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed))


def qini_curve(score: np.ndarray, treated: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    """Cumulative incremental outcomes when targeting customers in score order.

    At each cut, incremental = Y_treated - Y_control x (N_treated / N_control):
    the treated outcomes minus what those treated customers would have
    produced at the control rate observed in the same top slice.
    """
    order = np.argsort(-score, kind="stable")
    t = treated[order].astype(float)
    yy = y[order].astype(float)
    n_t = np.cumsum(t)
    n_c = np.cumsum(1 - t)
    y_t = np.cumsum(yy * t)
    y_c = np.cumsum(yy * (1 - t))
    with np.errstate(divide="ignore", invalid="ignore"):
        incremental = np.where(n_c > 0, y_t - y_c * n_t / n_c, 0.0)
    n = len(score)
    curve = pd.DataFrame({"fraction_targeted": np.arange(1, n + 1) / n, "incremental": incremental})
    return pd.concat(
        [pd.DataFrame({"fraction_targeted": [0.0], "incremental": [0.0]}), curve],
        ignore_index=True,
    )


def qini_coefficient(curve: pd.DataFrame) -> float:
    """Area between the model's Qini curve and random targeting, per customer.

    Random targeting is the straight line from 0 to the total incremental
    outcome. Positive means the ranking beats random; zero means no better.
    """
    x = curve["fraction_targeted"].to_numpy()
    q = curve["incremental"].to_numpy()
    random_line = x * q[-1]
    return float(np.trapezoid(q - random_line, x))


def decile_table(
    score: np.ndarray, treated: np.ndarray, y: np.ndarray, n_bins: int
) -> pd.DataFrame:
    """Observed treatment-minus-control outcome rate within each score bin."""
    frame = pd.DataFrame({"score": score, "treated": treated, "y": y})
    # Rank first so ties cannot produce empty or unequal bins.
    ranks = frame["score"].rank(method="first", ascending=False)
    frame["bin"] = pd.qcut(ranks, n_bins, labels=False) + 1
    rows = []
    for b, g in frame.groupby("bin"):
        t, c = g[g["treated"] == 1], g[g["treated"] == 0]
        rows.append(
            {
                "bin": int(b),
                "mean_predicted_uplift": float(g["score"].mean()),
                "n_treated": len(t),
                "n_control": len(c),
                "treated_rate": float(t["y"].mean()) if len(t) else np.nan,
                "control_rate": float(c["y"].mean()) if len(c) else np.nan,
            }
        )
    out = pd.DataFrame(rows)
    out["observed_uplift"] = out["treated_rate"] - out["control_rate"]
    return out


def fit_t_learner(
    df: pd.DataFrame,
    control: str,
    treatment: str,
    outcome: str,
    test_size: float,
    seed: int,
    n_bins: int,
    top_fraction: float = 0.3,
) -> UpliftResult:
    sub = df[df["segment"].isin([control, treatment])].reset_index(drop=True)
    X = features(sub)
    y = sub[outcome].to_numpy()
    treated = (sub["segment"] == treatment).to_numpy().astype(int)
    # Stratify on arm x outcome so both splits keep the same mix of each.
    strata = treated * 2 + y
    idx_train, idx_test = train_test_split(
        np.arange(len(sub)), test_size=test_size, random_state=seed, stratify=strata
    )
    tr_t = idx_train[treated[idx_train] == 1]
    tr_c = idx_train[treated[idx_train] == 0]
    model_t = _model(seed).fit(X.iloc[tr_t], y[tr_t])
    model_c = _model(seed).fit(X.iloc[tr_c], y[tr_c])

    X_test = X.iloc[idx_test]
    score = model_t.predict_proba(X_test)[:, 1] - model_c.predict_proba(X_test)[:, 1]
    t_test, y_test = treated[idx_test], y[idx_test]

    curve = qini_curve(score, t_test, y_test)
    total = curve["incremental"].iloc[-1]
    cut = int(round(top_fraction * len(score)))
    top_share = float(curve["incremental"].iloc[cut] / total) if total > 0 else float("nan")
    return UpliftResult(
        treatment_arm=treatment,
        outcome=outcome,
        n_train=len(idx_train),
        n_test=len(idx_test),
        deciles=decile_table(score, t_test, y_test, n_bins),
        qini_curve=curve,
        qini_coefficient=qini_coefficient(curve),
        top_share_of_incremental=top_share,
        top_fraction=top_fraction,
    )
