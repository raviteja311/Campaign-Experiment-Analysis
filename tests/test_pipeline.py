"""End-to-end: analysis, report files and CLI, on synthetic data with a known winner."""

import dataclasses
import json

import pytest
from conftest import ARMS, make_experiment

from campaign_experiment import cli
from campaign_experiment.analysis import run_analysis
from campaign_experiment.data import add_features
from campaign_experiment.report import to_jsonable, write_outputs

CONTROL, MENS, WOMENS = ARMS


@pytest.fixture
def fast_cfg(cfg):
    """The real config with the slow parts scaled down."""
    return dataclasses.replace(
        cfg,
        bootstrap=dataclasses.replace(cfg.bootstrap, n_resamples=200),
        peeking=dataclasses.replace(cfg.peeking, n_simulations=200, n_per_arm=2000, n_looks=5),
    )


@pytest.fixture
def womens_wins():
    """Womens email lifts visits and order value; Mens does nothing."""
    raw = make_experiment(
        n_per_arm=6000,
        visit_rate={CONTROL: 0.12, MENS: 0.12, WOMENS: 0.20},
        conversion_given_visit=0.1,
        spend_lift={CONTROL: 0.0, MENS: 0.0, WOMENS: 30.0},
        seed=8,
    )
    return add_features(raw, ARMS)


@pytest.fixture
def results(fast_cfg, womens_wins):
    return run_analysis(fast_cfg, womens_wins)


def test_analysis_recommends_the_true_winner(results):
    assert results["validity"]["srm"]["passed"]
    assert results["decision"]["recommended_arm"] == WOMENS
    mens = next(d for d in results["decision"]["arms"] if d["arm"] == MENS)
    assert "primary_significant_positive" in mens["failed"]


def test_no_recommendation_when_nothing_works(fast_cfg):
    df = add_features(make_experiment(n_per_arm=4000, seed=9), ARMS)
    assert run_analysis(fast_cfg, df)["decision"]["recommended_arm"] is None


def test_outputs_written_and_valid(results, tmp_path):
    paths = write_outputs(results, tmp_path)
    data = json.loads(paths["results"].read_text(encoding="utf-8"))
    assert data["decision"]["recommended_arm"] == WOMENS
    memo = paths["memo"].read_text(encoding="utf-8")
    assert "send the Womens E-Mail" in memo
    report = paths["report"].read_text(encoding="utf-8")
    for heading in ("## 2. Validity checks", "## 9. Segment effects", "## 14. Deviations"):
        assert heading in report
    figures = sorted(p.name for p in (tmp_path / "figures").glob("*.png"))
    assert figures == ["peeking.png", "primary_effects.png", "segment_effects.png", "uplift.png"]


def test_outputs_are_reproducible(results, tmp_path):
    first = write_outputs(results, tmp_path / "a")
    second = write_outputs(results, tmp_path / "b")
    for key in ("results", "report", "memo"):
        assert first[key].read_bytes() == second[key].read_bytes()


def test_json_has_no_nan_or_infinity(results):
    text = json.dumps(to_jsonable(results), allow_nan=False)
    assert "NaN" not in text


def test_to_jsonable_handles_special_floats():
    assert to_jsonable({"a": float("nan"), "b": float("inf"), "c": (1, 2)}) == {
        "a": None,
        "b": None,
        "c": [1, 2],
    }


def test_cli_plan_runs(capsys):
    assert cli.main(["plan", "--n-per-arm", "1000"]) == 0
    assert "MDE" in capsys.readouterr().out


def test_cli_run_end_to_end(monkeypatch, tmp_path, fast_cfg, womens_wins, capsys):
    monkeypatch.setattr(cli, "load_config", lambda _path: fast_cfg)
    monkeypatch.setattr("campaign_experiment.data.load_dataset", lambda _cfg: womens_wins)
    assert cli.main(["run", "--output", str(tmp_path)]) == 0
    assert f"recommendation: {WOMENS}" in capsys.readouterr().out
    assert (tmp_path / "DECISION_MEMO.md").exists()


def test_cli_reports_bad_config_without_traceback(tmp_path):
    bad = tmp_path / "params.yaml"
    bad.write_text("data: {}\n", encoding="utf-8")
    assert cli.main(["--params", str(bad), "plan"]) == 1
