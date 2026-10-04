"""Static report figures (PNG), drawn from the results dictionary only.

Colours follow the entity, never the rank: the Mens email is always blue and
the Womens email always orange, in every chart. The palette was checked for
colour-vision-deficiency separation; every chart also has a table in
REPORT.md, so no number is carried by colour alone.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # file output only; no display needed (CI, servers)
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_MUTED = "#52514e"
GRID = "#e4e3df"
NEUTRAL = "#8a8984"
ARM_COLORS = {"Mens E-Mail": "#2a78d6", "Womens E-Mail": "#eb6834"}
SERIES_A, SERIES_B = "#4a3aa7", "#1baf7a"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT_MUTED,
        "axes.titlecolor": TEXT,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": TEXT_MUTED,
        "ytick.color": TEXT_MUTED,
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    }
)


def _signed_usd(x: float) -> str:
    return f"{'+' if x > 0 else '-' if x < 0 else ''}${abs(x):.2f}"


def _zero_line(ax) -> None:
    ax.axvline(0, color=NEUTRAL, linewidth=1, zorder=1)


def _interval_row(ax, y, est, low, high, color, filled=True, label=None) -> None:
    ax.hlines(y, low, high, color=color, linewidth=2, zorder=2)
    ax.plot(
        est,
        y,
        "o",
        markersize=8,
        color=color,
        markerfacecolor=color if filled else SURFACE,
        markeredgewidth=2,
        markeredgecolor=SURFACE if filled else color,
        zorder=3,
        label=label,
    )


def primary_effects(results: dict[str, Any], path: Path) -> Path:
    """Lift versus control with 95% CIs: revenue, conversion and visits, one panel each.

    Three panels rather than one dual-axis chart: the metrics have different
    units, so each gets its own scale.
    """
    panels = [
        ("Revenue per customer ($)", results["primary"], 1.0, _signed_usd),
        (
            "Conversion rate (pp)",
            [r for r in results["secondary"] if r["metric"] == "conversion"],
            100.0,
            "{:+.2f} pp".format,
        ),
        (
            "Visit rate (pp)",
            [r for r in results["secondary"] if r["metric"] == "visit"],
            100.0,
            "{:+.1f} pp".format,
        ),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11, 2.6))
    for ax, (title, rows, scale, fmt) in zip(axes, panels, strict=True):
        for y, row in enumerate(reversed(rows)):
            color = ARM_COLORS[row["arm"]]
            _interval_row(
                ax,
                y,
                row["diff"] * scale,
                row["ci_low"] * scale,
                row["ci_high"] * scale,
                color,
            )
            ax.annotate(
                fmt(row["diff"] * scale),
                (row["ci_high"] * scale, y),
                xytext=(6, 0),
                textcoords="offset points",
                va="center",
                color=TEXT,
            )
        ax.set_yticks(range(len(rows)), [r["arm"] for r in reversed(rows)])
        ax.set_ylim(-0.6, len(rows) - 0.4)
        ax.set_title(title, loc="left")
        _zero_line(ax)
        ax.grid(axis="y", visible=False)
        left, right = ax.get_xlim()
        ax.set_xlim(min(left, 0), right + (right - left) * 0.25)
    fig.suptitle(
        "Lift versus no email, 95% confidence intervals",
        x=0.01,
        ha="left",
        fontsize=12,
        fontweight="bold",
        color=TEXT,
    )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def segment_effects(results: dict[str, Any], path: Path) -> Path:
    """Revenue lift within every segment level. Filled = Holm-significant."""
    arms = list(results["segments"])
    first = results["segments"][arms[0]]
    labels = [f"{r['segment']}: {r['level']}" for r in first]
    fig, axes = plt.subplots(1, len(arms), figsize=(10, 0.32 * len(labels) + 1.2), sharey=True)
    for ax, arm in zip(axes, arms, strict=True):
        rows = {f"{r['segment']}: {r['level']}": r for r in results["segments"][arm]}
        for y, label in enumerate(reversed(labels)):
            r = rows[label]
            _interval_row(
                ax, y, r["diff"], r["ci_low"], r["ci_high"], ARM_COLORS[arm], r["significant"]
            )
        overall = next(p for p in results["primary"] if p["arm"] == arm)["diff"]
        ax.axvline(overall, color=ARM_COLORS[arm], linewidth=1, linestyle=":", zorder=1)
        ax.set_title(arm, loc="left")
        ax.set_xlabel("Revenue lift per customer ($)")
        _zero_line(ax)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(labels)), list(reversed(labels)))
    fig.text(
        0.01,
        -0.02,
        "Filled marker: significant after Holm correction across all segments for that email. "
        "Hollow: not significant. Dotted line: overall lift.",
        color=TEXT_MUTED,
        fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def peeking(results: dict[str, Any], path: Path) -> Path:
    pk = results["peeking"]
    looks = pk["looks"]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.axhline(pk["alpha"], color=NEUTRAL, linewidth=1, linestyle="--")
    ax.annotate(
        f"nominal alpha {pk['alpha']:.0%}",
        (looks[-1], pk["alpha"]),
        xytext=(0, 5),
        textcoords="offset points",
        ha="right",
        color=TEXT_MUTED,
    )
    for values, color, label, offset in (
        (pk["false_positive_rate"], SERIES_A, "Stop at first p < 0.05", (-10, 6)),
        (pk["bonferroni_false_positive_rate"], SERIES_B, "Bonferroni across looks", (-10, -16)),
    ):
        ax.plot(looks, values, "-o", color=color, linewidth=2, markersize=8, label=label)
        ax.annotate(
            f"{values[-1]:.1%}",
            (looks[-1], values[-1]),
            xytext=offset,
            textcoords="offset points",
            ha="right",
            color=TEXT,
        )
    ax.set_xscale("log")
    ax.set_xticks(looks, [str(m) for m in looks])
    ax.minorticks_off()
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_ylim(0, max(pk["false_positive_rate"]) * 1.2)
    ax.set_xlabel("Number of looks at the running test")
    ax.set_ylabel("False-positive rate (A/A)")
    ax.set_title(
        f"Peeking inflates false positives ({pk['n_simulations']:,} simulated A/A tests)",
        loc="left",
    )
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def uplift(results: dict[str, Any], path: Path) -> Path:
    up = results["uplift"]
    color = ARM_COLORS.get(up["treatment_arm"], SERIES_A)
    deciles = up["deciles"]
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 3.4))

    bins = [d["bin"] for d in deciles]
    observed = [d["observed_uplift"] * 100 for d in deciles]
    left.bar(bins, observed, color=color, width=0.8, edgecolor=SURFACE, linewidth=2)
    left.axhline(0, color=NEUTRAL, linewidth=1)
    left.set_xticks(bins)
    left.set_xlabel("Predicted-uplift decile (1 = highest)")
    left.set_ylabel(f"Observed {up['outcome']} uplift (pp)")
    left.set_title("Held-out customers, ranked by the model", loc="left")
    left.grid(axis="x", visible=False)

    curve = up["qini_curve"]
    x = [p["fraction_targeted"] for p in curve]
    q = [p["incremental"] for p in curve]
    right.plot(x, q, color=color, linewidth=2, label="T-learner ranking")
    right.plot([0, 1], [0, q[-1]], color=NEUTRAL, linewidth=1, linestyle="--", label="Random")
    cut = up["top_fraction"]
    right.axvline(cut, color=NEUTRAL, linewidth=1, linestyle=":", zorder=0)
    right.annotate(
        f"top {cut:.0%} targeted: {up['top_share_of_incremental']:.0%} "
        f"of incremental {up['outcome']}s",
        (cut, max(q) * 0.05),
        xytext=(6, 0),
        textcoords="offset points",
        color=TEXT,
    )
    right.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    right.set_xlabel("Share of customers targeted")
    right.set_ylabel(f"Incremental {up['outcome']}s")
    right.set_title(f"Qini curve (coefficient {up['qini_coefficient']:.2f})", loc="left")
    right.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def render_all(results: dict[str, Any], out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    return [
        primary_effects(results, out_dir / "primary_effects.png"),
        segment_effects(results, out_dir / "segment_effects.png"),
        peeking(results, out_dir / "peeking.png"),
        uplift(results, out_dir / "uplift.png"),
    ]
