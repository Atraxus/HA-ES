"""Test Elo against inference time for the ensembles of the pruned pools."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

POOL_COLORS = {"cross_family": "#2a78d6", "across_foundation_models": "#eb6834"}
PARETO_FRONT_COLOR = "#8a8984"

POOL_LABELS = {
    "cross_family": "Cross-family [pruned]",
    "across_foundation_models": "Across foundation models [pruned]",
}
ALGORITHM_LABELS = {
    "mo_ges": "MO-GES",
    "mo_ges_archive": "MO-GES (archive)",
    "infer_qdo": "Infer-QDO-ES",
    "infer_qdo_3_batches": "Infer-QDO-ES (3 batches)",
    "infer_qdo_sparse": "Infer-QDO-ES (sparse, dynamic)",
    "infer_qdo_sparse_p20": "Infer-QDO-ES (sparse, static 20%)",
    "infer_qdo_sparse_p50": "Infer-QDO-ES (sparse, static 50%)",
}
# One family's single model, drawn and labelled point by point.
FAMILY = "family"
VISIBLE_ELO = 1400.0
# Searches with one ensemble per task, drawn as single points.
BASELINE_LABELS = {"ges": "GES", "qoes": "QO-ES", "qdoes": "QDO-ES"}
_MARKERS = {
    "mo_ges": "o",
    "mo_ges_archive": ">",
    "infer_qdo": "P",
    "infer_qdo_3_batches": "h",
    "infer_qdo_sparse": "X",
    "infer_qdo_sparse_p20": "s",
    "infer_qdo_sparse_p50": "<",
    "ges": "D",
    "qoes": "^",
    "qdoes": "v",
}
_LINE_STYLES = {
    "mo_ges": "-",
    "mo_ges_archive": (0, (5, 1)),
    "infer_qdo": ":",
    "infer_qdo_3_batches": (0, (4, 1, 1, 1, 1, 1)),
    "infer_qdo_sparse": "-.",
    "infer_qdo_sparse_p20": "--",
    "infer_qdo_sparse_p50": (0, (3, 1, 1, 1)),
}
_FAMILY_MARKER = "*"
_TIME = "time_infer_s_per_1K"
# Direct pool labels beside one MO-GES point: (point, x offset, y offset, alignment), in points.
_POOL_LABEL_PLACEMENT = {
    "cross_family": (0, 8, -14, "left"),
    "across_foundation_models": (4, -6, 26, "right"),
}
# Without MO-GES the first line is Infer-QDO-ES, which climbs into the upper corner: label below it.
_POOL_LABEL_PLACEMENT_NO_MO_GES = {
    "cross_family": (0, 8, -14, "left"),
    "across_foundation_models": (0, 6, -34, "left"),
}
# Family labels sit above their star, except where that would cross a line or another label.
_FAMILY_LABEL_DEFAULT = (0, 7, "center")
_FAMILY_LABEL_PLACEMENT = {
    "TabFM": (0, -14, "center"),
    "CatBoost": (-6, 5, "right"),
    "TabICLv2": (6, -4, "left"),
}


def pareto_front(points: pd.DataFrame) -> pd.DataFrame:
    """The points no other beats on both time (lower) and Elo (higher), fastest first."""
    ordered = points.sort_values([_TIME, "elo"], ascending=[True, False], kind="stable")
    previous_best = ordered["elo"].cummax().shift(fill_value=-float("inf"))
    return ordered[ordered["elo"] > previous_best]


def mark_visible_families(points: pd.DataFrame, min_elo: float = VISIBLE_ELO) -> pd.DataFrame:
    """Add ``shown``: a family's model is drawn above ``min_elo`` or on the Pareto front."""
    is_family = points["algorithm"] == FAMILY
    on_front = points.index.isin(pareto_front(points).index)
    return points.assign(shown=~is_family | (points["elo"] > min_elo) | on_front)


def _legend_handles(points: pd.DataFrame) -> list[Line2D]:
    """Pool colors, then one neutral entry per method, so the legend stays short."""
    ink = "#52514e"
    handles = [
        Line2D([], [], marker="o", linestyle="none", color=POOL_COLORS[pool], label=label)
        for pool, label in POOL_LABELS.items()
    ]
    handles += [
        Line2D(
            [],
            [],
            marker=_MARKERS[algorithm],
            linestyle=_LINE_STYLES[algorithm],
            color=ink,
            linewidth=2,
            label=label,
        )
        for algorithm, label in ALGORITHM_LABELS.items()
        if algorithm in set(points.loc[points["position"].notna(), "algorithm"])
    ]
    shown = set(points.loc[points["algorithm"].isin(BASELINE_LABELS), "algorithm"])
    handles += [
        Line2D([], [], marker=_MARKERS[algorithm], linestyle="none", color=ink, label=label)
        for algorithm, label in BASELINE_LABELS.items()
        if algorithm in shown
    ]
    if (points["algorithm"] == FAMILY).any():
        handles.append(
            Line2D(
                [],
                [],
                marker=_FAMILY_MARKER,
                markersize=9,
                linestyle="none",
                color=ink,
                label="One family, best config on validation",
            )
        )
    handles += [
        Line2D(
            [],
            [],
            marker="o",
            markersize=12,
            markerfacecolor="none",
            markeredgecolor=ink,
            linestyle="none",
            label="Best ensemble on validation",
        ),
        Line2D([], [], color=PARETO_FRONT_COLOR, linewidth=1.5, label="Pareto front, all points"),
    ]
    return handles


def plot_inference_pareto(points: pd.DataFrame, path: Path) -> Path:
    """One line per (pool, algorithm), from the fastest ensemble to the best on validation.

    Color is the pool and marker and line style the algorithm. The large ring marks each line's
    best-validation ensemble, and the gray staircase is the Pareto front over all points.
    Baselines (no ``position``) have one ensemble per task and are single points.
    """
    with sns.axes_style("whitegrid"):
        fig, ax = plt.subplots(figsize=(11.5, 6))
    front = pareto_front(points)
    ax.step(
        front[_TIME],
        front["elo"],
        where="post",
        color=PARETO_FRONT_COLOR,
        linewidth=1.5,
        zorder=1,
    )
    if "shown" in points:
        points = points[points["shown"]]
    families = points[points["algorithm"] == FAMILY]
    for row in families.itertuples(index=False):
        ax.scatter(
            row.time_infer_s_per_1K,
            row.elo,
            s=55,
            marker=_FAMILY_MARKER,
            color=POOL_COLORS[row.pool],
            edgecolors="#0b0b0b",
            linewidths=0.6,
            zorder=4,
        )
        name = row.entrant.rsplit(" (", 1)[0]
        dx, dy, alignment = _FAMILY_LABEL_PLACEMENT.get(name, _FAMILY_LABEL_DEFAULT)
        ax.annotate(
            name,
            (row.time_infer_s_per_1K, row.elo),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=alignment,
            fontsize=7.5,
            color="#0b0b0b",
            zorder=5,
        )
    baselines = points[points["algorithm"].isin(BASELINE_LABELS)]
    for row in baselines.itertuples(index=False):
        ax.scatter(
            row.time_infer_s_per_1K,
            row.elo,
            s=75,
            marker=_MARKERS[row.algorithm],
            color=POOL_COLORS[row.pool],
            edgecolors="#0b0b0b",
            linewidths=0.8,
            zorder=4,
        )
    lines = points[points["position"].notna()]
    for (pool, algorithm), line in lines.groupby(["pool", "algorithm"], sort=False):
        line = line.sort_values("position")
        color = POOL_COLORS[pool]
        ax.plot(
            line[_TIME],
            line["elo"],
            color=color,
            linestyle=_LINE_STYLES[algorithm],
            linewidth=2,
            marker=_MARKERS[algorithm],
            markersize=7,
            markeredgecolor="white",
            markeredgewidth=0.6,
            zorder=2,
        )
        if algorithm == lines["algorithm"].iloc[0]:
            placement = (
                _POOL_LABEL_PLACEMENT
                if algorithm.startswith("mo_ges")
                else _POOL_LABEL_PLACEMENT_NO_MO_GES
            )
            index, dx, dy, alignment = placement[pool]
            first = line.iloc[index if len(line) > index else 0]
            ax.annotate(
                POOL_LABELS[pool].replace("foundation models", "foundation\nmodels"),
                (first[_TIME], first["elo"]),
                xytext=(dx, dy),
                textcoords="offset points",
                ha=alignment,
                fontsize=10,
                color="#0b0b0b",
            )
        best = line.iloc[-1]
        ax.scatter(
            best[_TIME],
            best["elo"],
            s=170,
            facecolors="none",
            edgecolors=color,
            linewidths=1.5,
            zorder=3,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Ensemble inference time per 1K rows, mean over datasets (s, log scale)")
    ax.set_ylabel("Test Elo (RF default = 1000)")
    # Outside the axes: both pools' points end near the lower right, where a legend would cover them.
    ax.legend(
        handles=_legend_handles(points),
        frameon=True,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        fontsize=9,
    )
    ax.set_title("Inference time against test performance of the ensembles found")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path
