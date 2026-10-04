"""Heterogeneous effects: the revenue effect within each segment level."""

from __future__ import annotations

import pandas as pd

from campaign_experiment.stats import holm, welch_test


def segment_effects(seg: pd.DataFrame, control: str, treatment: str, alpha: float) -> pd.DataFrame:
    """Welch test per (segment, level), Holm-corrected across all of them.

    `seg` is the output of metrics_sql.segment_metrics. All levels of all
    segments for one treatment form one family: a reader scanning the table
    for "where does it work" is running every one of these tests.
    """
    rows = []
    for (segment, level), group in seg.groupby(["segment", "level"], sort=True):
        by_arm = group.set_index("arm")
        if control not in by_arm.index or treatment not in by_arm.index:
            continue
        c, t = by_arm.loc[control], by_arm.loc[treatment]
        if c["n"] < 2 or t["n"] < 2:
            continue
        est = welch_test(
            c["revenue_per_customer"],
            c["revenue_var"],
            int(c["n"]),
            t["revenue_per_customer"],
            t["revenue_var"],
            int(t["n"]),
            alpha,
            "spend",
        )
        rows.append(
            {
                "segment": segment,
                "level": level,
                "n_control": est.n_control,
                "n_treatment": est.n_treatment,
                "control_mean": est.control_mean,
                "treatment_mean": est.treatment_mean,
                "diff": est.diff,
                "ci_low": est.ci_low,
                "ci_high": est.ci_high,
                "p_value": est.p_value,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["p_holm"] = holm(out["p_value"].tolist())
    out["significant"] = out["p_holm"] < alpha
    out["harm"] = out["significant"] & (out["diff"] < 0)
    return out
