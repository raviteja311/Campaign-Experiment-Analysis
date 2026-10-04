import dataclasses
import gzip
import hashlib
from pathlib import Path

import pandas as pd
import pytest
from conftest import ARMS

from campaign_experiment import data
from campaign_experiment.data import (
    DataDownloadError,
    DataValidationError,
    add_features,
    ensure_dataset,
    load_dataset,
    validate,
)


def _with_sources(cfg, tmp_path, primary: Path, mirror: Path, primary_sha=None, mirror_sha=None):
    """A config whose two sources are local file:// URLs and whose cache is tmp_path."""
    d = dataclasses.replace(
        cfg.data,
        primary_url=primary.as_uri(),
        primary_sha256=primary_sha or hashlib.sha256(primary.read_bytes()).hexdigest(),
        mirror_url=mirror.as_uri(),
        mirror_sha256=mirror_sha or hashlib.sha256(mirror.read_bytes()).hexdigest(),
        cache_dir=str(tmp_path / "cache"),
        retries=1,
    )
    return dataclasses.replace(cfg, data=d)


@pytest.fixture
def sources(tmp_path, raw_frame):
    primary = tmp_path / "primary.csv"
    raw_frame.to_csv(primary, index=False)
    mirror = tmp_path / "mirror.csv.gz"
    with gzip.open(mirror, "wt") as fh:
        raw_frame.to_csv(fh, index=False)
    return primary, mirror


# --- validation -----------------------------------------------------------


def test_valid_frame_passes(raw_frame):
    assert validate(raw_frame, ARMS) is raw_frame


def test_missing_column_reported(raw_frame):
    with pytest.raises(DataValidationError, match="missing columns: \\['spend'\\]"):
        validate(raw_frame.drop(columns="spend"), ARMS)


def test_every_problem_collected_before_raising(raw_frame):
    bad = raw_frame.copy()
    bad.loc[0, "visit"] = 2
    bad.loc[1, "recency"] = 13
    with pytest.raises(DataValidationError) as info:
        validate(bad, ARMS)
    assert len(info.value.problems) >= 2


def test_conversion_without_visit_rejected(raw_frame):
    bad = raw_frame.copy()
    row = bad.index[bad["visit"] == 0][0]
    bad.loc[row, ["conversion", "spend"]] = [1, 50.0]
    with pytest.raises(DataValidationError, match="conversion without a visit"):
        validate(bad, ARMS)


def test_spend_without_conversion_rejected(raw_frame):
    bad = raw_frame.copy()
    row = bad.index[bad["conversion"] == 0][0]
    bad.loc[row, "spend"] = 10.0
    with pytest.raises(DataValidationError, match="spend > 0"):
        validate(bad, ARMS)


def test_unknown_arm_rejected(raw_frame):
    bad = raw_frame.copy()
    bad.loc[0, "segment"] = "SMS"
    with pytest.raises(DataValidationError, match="unknown arms"):
        validate(bad, ARMS)


def test_history_segment_must_match_history(raw_frame):
    bad = raw_frame.copy()
    bad.loc[0, ["history", "history_segment"]] = [50.0, "7) $1,000 +"]
    with pytest.raises(DataValidationError, match="history_segment disagrees"):
        validate(bad, ARMS)


def test_nulls_reported(raw_frame):
    bad = raw_frame.copy()
    bad.loc[0, "channel"] = None
    with pytest.raises(DataValidationError, match="channel: 1 missing"):
        validate(bad, ARMS)


def test_empty_frame_rejected(raw_frame):
    with pytest.raises(DataValidationError, match="no rows"):
        validate(raw_frame.iloc[0:0], ARMS)


def test_duplicate_rows_are_kept(raw_frame):
    doubled = raw_frame.iloc[[0, 0]]
    padded = validate(pd.concat([raw_frame, doubled]), ARMS)
    assert len(padded) == len(raw_frame) + 2


def test_features(frame):
    assert list(frame["arm"].cat.categories) == ARMS
    assert set(frame["purchase_line"]) <= {"mens_only", "womens_only", "both"}
    both = frame["purchase_line"] == "both"
    assert ((frame.loc[both, "mens"] == 1) & (frame.loc[both, "womens"] == 1)).all()


# --- acquisition ----------------------------------------------------------


def test_downloads_and_verifies_primary(cfg, tmp_path, sources):
    local = _with_sources(cfg, tmp_path, *sources)
    path = ensure_dataset(local)
    assert path.name == "hillstrom.csv"
    assert not list(path.parent.glob("*.part"))


def test_cached_copy_reused_without_download(cfg, tmp_path, sources, monkeypatch):
    local = _with_sources(cfg, tmp_path, *sources)
    ensure_dataset(local)

    def no_network(*args, **kwargs):
        raise AssertionError("should not download when a verified copy is cached")

    monkeypatch.setattr(data, "_fetch", no_network)
    assert ensure_dataset(local).name == "hillstrom.csv"


def test_corrupted_cache_is_replaced(cfg, tmp_path, sources):
    local = _with_sources(cfg, tmp_path, *sources)
    path = ensure_dataset(local)
    path.write_text("tampered", encoding="utf-8")
    assert data.sha256_of(ensure_dataset(local)) == local.data.primary_sha256


def test_falls_back_to_mirror_on_checksum_mismatch(cfg, tmp_path, sources):
    local = _with_sources(cfg, tmp_path, *sources, primary_sha="0" * 64)
    path = ensure_dataset(local)
    assert path.name == "hillstrom_mirror.csv.gz"
    assert not (path.parent / "hillstrom.csv").exists()


def test_all_sources_failing_raises(cfg, tmp_path, sources):
    local = _with_sources(cfg, tmp_path, *sources, primary_sha="0" * 64, mirror_sha="1" * 64)
    with pytest.raises(DataDownloadError, match="no source produced a verified file"):
        ensure_dataset(local)


def test_unreachable_source_falls_back(cfg, tmp_path, sources):
    local = _with_sources(cfg, tmp_path, *sources)
    local = dataclasses.replace(
        local, data=dataclasses.replace(local.data, primary_url=(tmp_path / "gone.csv").as_uri())
    )
    assert ensure_dataset(local).name == "hillstrom_mirror.csv.gz"


def test_gzipped_mirror_loads_end_to_end(cfg, tmp_path, sources, raw_frame):
    local = _with_sources(cfg, tmp_path, *sources, primary_sha="0" * 64)
    df = load_dataset(local)
    assert len(df) == len(raw_frame)
    assert "purchase_line" in df.columns


@pytest.mark.network
def test_real_dataset(cfg):
    """Downloads the real file (4 MB). Run with `pytest -m network`."""
    df = load_dataset(cfg)
    assert len(df) == 64_000
    assert df["segment"].value_counts().to_dict() == {
        "Womens E-Mail": 21_387,
        "Mens E-Mail": 21_307,
        "No E-Mail": 21_306,
    }
    add_features(df, cfg.experiment.arms)
