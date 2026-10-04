from pathlib import Path

import pytest
import yaml

from campaign_experiment.config import PARAMS_PATH, ConfigError, load_config


def _write(tmp_path: Path, mutate) -> Path:
    raw = yaml.safe_load(PARAMS_PATH.read_text(encoding="utf-8"))
    mutate(raw)
    path = tmp_path / "params.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


def test_committed_params_load():
    cfg = load_config()
    assert cfg.experiment.arms[0] == cfg.experiment.control_arm
    assert len(cfg.experiment.arms) == 3


def test_unknown_key_rejected(tmp_path):
    path = _write(tmp_path, lambda r: r["experiment"].update(alhpa=0.05))
    with pytest.raises(ConfigError, match="unknown keys"):
        load_config(path)


def test_missing_section_rejected(tmp_path):
    path = _write(tmp_path, lambda r: r.pop("bootstrap"))
    with pytest.raises(ConfigError, match="missing section 'bootstrap'"):
        load_config(path)


def test_allocation_must_sum_to_one(tmp_path):
    path = _write(tmp_path, lambda r: r["experiment"].update(expected_allocation=[0.5, 0.3, 0.3]))
    with pytest.raises(ConfigError, match="sum to 1"):
        load_config(path)


def test_uplift_arm_must_be_a_treatment(tmp_path):
    path = _write(tmp_path, lambda r: r["uplift"].update(treatment_arm="No E-Mail"))
    with pytest.raises(ConfigError, match="treatment_arms"):
        load_config(path)


def test_invalid_alpha_rejected(tmp_path):
    path = _write(tmp_path, lambda r: r["experiment"].update(alpha=1.5))
    with pytest.raises(ConfigError, match="alpha"):
        load_config(path)
