import pandas as pd
import pytest

from campaign_experiment.metrics_sql import arm_metrics, connect, segment_metrics


@pytest.fixture
def con(frame):
    return connect(frame)


def test_arm_metrics_match_pandas(con, frame):
    sql = arm_metrics(con)
    pdf = frame.groupby("segment").agg(
        n=("spend", "size"),
        revenue_per_customer=("spend", "mean"),
        revenue_var=("spend", "var"),
        conversion_rate=("conversion", "mean"),
        visit_rate=("visit", "mean"),
    )
    for col in pdf.columns:
        pd.testing.assert_series_equal(
            sql[col].sort_index().astype(float),
            pdf[col].sort_index().astype(float),
            check_names=False,
            check_index_type=False,
        )


def test_window_shares_sum_to_one(con):
    sql = arm_metrics(con)
    assert sql["share_of_customers"].sum() == pytest.approx(1.0)
    assert sql["share_of_revenue"].sum() == pytest.approx(1.0)


def test_order_value_only_counts_converters(con, frame):
    sql = arm_metrics(con)
    arm = sql.index[0]
    converters = frame[(frame["segment"] == arm) & (frame["conversion"] == 1)]
    assert sql.loc[arm, "order_value"] == pytest.approx(converters["spend"].mean())


def test_segment_metrics_cover_every_level_and_arm(con, frame):
    seg = segment_metrics(con, ["channel", "newbie"])
    assert set(seg["segment"]) == {"channel", "newbie"}
    channel = seg[seg["segment"] == "channel"]
    assert channel["n"].sum() == len(frame)
    assert set(channel["level"]) == set(frame["channel"])
    # Arm shares within a level sum to one (the window partition).
    sums = seg.groupby(["segment", "level"])["arm_share"].sum()
    assert (sums.round(9) == 1).all()


def test_segment_metric_matches_pandas(con, frame):
    seg = segment_metrics(con, ["channel"])
    row = seg[(seg["level"] == "Web") & (seg["arm"] == "Womens E-Mail")].iloc[0]
    sub = frame[(frame["channel"] == "Web") & (frame["segment"] == "Womens E-Mail")]
    assert row["n"] == len(sub)
    assert row["revenue_per_customer"] == pytest.approx(sub["spend"].mean())


@pytest.mark.parametrize("segments", [[], ["spend"], ["channel; DROP TABLE customers"]])
def test_segment_columns_whitelisted(con, segments):
    with pytest.raises(ValueError, match="subset"):
        segment_metrics(con, segments)
