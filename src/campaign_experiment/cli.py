"""Command line entry point: `campaign-experiment <command>` or `python -m campaign_experiment`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from campaign_experiment.config import PARAMS_PATH, ConfigError, load_config
from campaign_experiment.data import DataDownloadError, DataValidationError
from campaign_experiment.logger import get_logger

log = get_logger("campaign_experiment.cli")

# Planning grids from docs/TEST_PLAN.md, section 7: assumed baselines, since
# the outcomes are not known when a test is planned.
PLAN_RATES = {"visit": [0.10, 0.15, 0.20], "conversion": [0.005, 0.01, 0.02]}
PLAN_SPEND_SD = [10.0, 15.0, 20.0]


def cmd_download(args: argparse.Namespace) -> int:
    from campaign_experiment.data import ensure_dataset

    path = ensure_dataset(load_config(args.params))
    print(f"verified dataset at {path}")
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    from campaign_experiment.power import mde_mean, mde_proportion

    cfg = load_config(args.params)
    exp = cfg.experiment
    alpha = exp.alpha / len(exp.treatment_arms)
    n = args.n_per_arm
    print(f"MDE at n={n:,} per arm, alpha={alpha:.4f} per comparison, power={exp.power:.0%}")
    for metric, rates in PLAN_RATES.items():
        for p in rates:
            mde = mde_proportion(p, n, n, alpha, exp.power)
            print(f"  {metric:<10} baseline {p:>6.1%}  MDE {mde * 100:.3f} pp")
    for sd in PLAN_SPEND_SD:
        print(f"  {'spend':<10} sd ${sd:>5.2f}  MDE ${mde_mean(sd, n, n, alpha, exp.power):.3f}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from campaign_experiment.analysis import run_analysis
    from campaign_experiment.data import load_dataset
    from campaign_experiment.report import write_outputs

    cfg = load_config(args.params)
    out_dir = Path(args.output) if args.output else cfg.resolve(cfg.report.output_dir)
    results = run_analysis(cfg, load_dataset(cfg))
    paths = write_outputs(results, out_dir)
    rec = results["decision"]["recommended_arm"] or "neither email"
    print(f"recommendation: {rec}")
    for name, path in paths.items():
        print(f"{name:<8} {path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="campaign-experiment",
        description="Pre-registered analysis of the Hillstrom email campaign experiment.",
    )
    parser.add_argument("--params", default=str(PARAMS_PATH), help="path to params.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("download", help="download and verify the dataset").set_defaults(
        func=cmd_download
    )

    plan = sub.add_parser("plan", help="print the planning MDE grid (no outcome data used)")
    plan.add_argument("--n-per-arm", type=int, default=21_300)
    plan.set_defaults(func=cmd_plan)

    run = sub.add_parser("run", help="run the full analysis and write the report")
    run.add_argument("--output", help="output directory (default: report.output_dir)")
    run.set_defaults(func=cmd_run)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (ConfigError, DataDownloadError, DataValidationError) as exc:
        # Expected failures: a clear message, no traceback, non-zero exit.
        log.error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
