"""Load params.yaml into typed, immutable settings.

Every module receives its settings from here, so a value is defined once and
a typo in params.yaml fails at load time rather than halfway through a run.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARAMS_PATH = PROJECT_ROOT / "params.yaml"


class ConfigError(ValueError):
    """params.yaml is missing a section, has an unknown key, or an invalid value."""


@dataclass(frozen=True)
class DataConfig:
    primary_url: str
    primary_sha256: str
    mirror_url: str
    mirror_sha256: str
    cache_dir: str
    timeout_seconds: int
    retries: int


@dataclass(frozen=True)
class ExperimentConfig:
    control_arm: str
    treatment_arms: list[str]
    expected_allocation: list[float]
    primary_metric: str
    secondary_metrics: list[str]
    alpha: float
    power: float
    srm_alpha: float
    balance_smd_threshold: float
    cuped_covariate: str
    segments: list[str]
    winsorize_quantile: float

    @property
    def arms(self) -> list[str]:
        """Control first, then treatments: the order every table uses."""
        return [self.control_arm, *self.treatment_arms]


@dataclass(frozen=True)
class BootstrapConfig:
    n_resamples: int
    confidence: float
    seed: int


@dataclass(frozen=True)
class EconomicsConfig:
    gross_margin: float
    cost_per_email: float
    rollout_customers: int


@dataclass(frozen=True)
class PeekingConfig:
    n_simulations: int
    n_per_arm: int
    n_looks: int
    seed: int


@dataclass(frozen=True)
class UpliftConfig:
    treatment_arm: str
    outcome: str
    test_size: float
    seed: int
    n_bins: int


@dataclass(frozen=True)
class ReportConfig:
    output_dir: str


@dataclass(frozen=True)
class Config:
    data: DataConfig
    experiment: ExperimentConfig
    bootstrap: BootstrapConfig
    economics: EconomicsConfig
    peeking: PeekingConfig
    uplift: UpliftConfig
    report: ReportConfig

    def resolve(self, relative: str) -> Path:
        """Turn a path from params.yaml into an absolute path under the project."""
        path = Path(relative)
        return path if path.is_absolute() else PROJECT_ROOT / path


def _section(raw: dict[str, Any], name: str, cls: type) -> Any:
    if name not in raw or not isinstance(raw[name], dict):
        raise ConfigError(f"params.yaml: missing section '{name}'")
    values = raw[name]
    expected = {f.name for f in fields(cls)}
    unknown = set(values) - expected
    missing = expected - set(values)
    if unknown:
        raise ConfigError(f"params.yaml [{name}]: unknown keys {sorted(unknown)}")
    if missing:
        raise ConfigError(f"params.yaml [{name}]: missing keys {sorted(missing)}")
    return cls(**values)


def _validate(cfg: Config) -> None:
    exp = cfg.experiment
    if not 0 < exp.alpha < 1 or not 0 < exp.srm_alpha < 1:
        raise ConfigError("alpha and srm_alpha must be in (0, 1)")
    if not 0 < exp.power < 1:
        raise ConfigError("power must be in (0, 1)")
    if len(exp.expected_allocation) != len(exp.arms):
        raise ConfigError("expected_allocation needs one share per arm (control first)")
    if abs(sum(exp.expected_allocation) - 1) > 1e-6:
        raise ConfigError("expected_allocation must sum to 1")
    if len(set(exp.arms)) != len(exp.arms):
        raise ConfigError("arm names must be unique")
    if not 0.5 < exp.winsorize_quantile < 1:
        raise ConfigError("winsorize_quantile must be in (0.5, 1)")
    if not 0 < cfg.bootstrap.confidence < 1 or cfg.bootstrap.n_resamples < 100:
        raise ConfigError("bootstrap needs confidence in (0, 1) and at least 100 resamples")
    if not 0 < cfg.economics.gross_margin <= 1 or cfg.economics.cost_per_email < 0:
        raise ConfigError("gross_margin must be in (0, 1] and cost_per_email >= 0")
    if cfg.peeking.n_looks < 2:
        raise ConfigError("peeking.n_looks must be at least 2")
    if cfg.uplift.treatment_arm not in exp.treatment_arms:
        raise ConfigError("uplift.treatment_arm must be one of experiment.treatment_arms")
    if not 0 < cfg.uplift.test_size < 1:
        raise ConfigError("uplift.test_size must be in (0, 1)")


def load_config(path: Path | str = PARAMS_PATH) -> Config:
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ConfigError(f"{path} is empty or not a mapping")
    cfg = Config(
        data=_section(raw, "data", DataConfig),
        experiment=_section(raw, "experiment", ExperimentConfig),
        bootstrap=_section(raw, "bootstrap", BootstrapConfig),
        economics=_section(raw, "economics", EconomicsConfig),
        peeking=_section(raw, "peeking", PeekingConfig),
        uplift=_section(raw, "uplift", UpliftConfig),
        report=_section(raw, "report", ReportConfig),
    )
    _validate(cfg)
    return cfg
