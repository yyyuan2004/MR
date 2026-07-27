#!/usr/bin/env python3
"""Phase diagram: where does a nonlinear/learned prior actually pay off?

x = acceleration factor, y = mask coherence, color = PSNR gain of a
prior-based reconstruction over zero-filling. The color scale is diverging
with its neutral midpoint pinned exactly at zero, and the zero level is drawn
as an explicit contour, so the sign-flip boundary — the regime where the
prior stops helping and starts hurting — is readable without relying on color
alone.

Consumes runs/<experiment_name>/metrics/budget_sweep.csv (scripts/11).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mrsim.config import load_config, run_dir

# Diverging pair with a neutral midpoint (no rainbow, no hue at the midpoint).
DIVERGING = "RdBu_r"


def family_of(mask_name: str) -> str:
    """Coarse family label, used as a secondary (non-color) encoding."""
    for prefix, label in (
        ("uniform_random", "uniform"),
        ("variable_density_lines", "vd lines"),
        ("variable_density", "variable density"),
        ("multilevel", "multilevel"),
        ("equispaced", "equispaced"),
        ("psf_penalized", "psf-penalized"),
        ("aopt_greedy", "A-optimal"),
        ("line_aopt", "line A-opt"),
        ("subspace_aopt", "subspace A-opt"),
        ("spectrum_energy", "spectrum energy"),
        ("line_subspace", "subspace leakage"),
    ):
        if mask_name.startswith(prefix):
            return label
    return "other"


MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">", "h"]


def panel(ax, table: pd.DataFrame, gain_col: str, y_col: str, norm, families: list[str]) -> None:
    x = table["acceleration"].to_numpy(dtype=float)
    y = table[y_col].to_numpy(dtype=float)
    gain = table[gain_col].to_numpy(dtype=float)

    # Field behind the points. Only three acceleration values are measured, so
    # this interpolates across gaps: it conveys the trend, the markers are the
    # data. The zero level is the headline and is drawn on top.
    if len(np.unique(x)) >= 2 and len(np.unique(y)) >= 3:
        try:
            tri = mtri.Triangulation(x, y)
            ax.tricontourf(tri, gain, levels=20, cmap=DIVERGING, norm=norm, alpha=0.5)
            if gain.min() < 0.0 < gain.max():
                ax.tricontour(tri, gain, levels=[0.0], colors="#111111", linewidths=2.0)
        except (ValueError, RuntimeError):
            pass

    for family in families:
        block = table[table["mask"].map(family_of) == family]
        if not len(block):
            continue
        ax.scatter(
            block["acceleration"], block[y_col], c=block[gain_col],
            cmap=DIVERGING, norm=norm, s=88,
            marker=MARKERS[families.index(family) % len(MARKERS)],
            edgecolors="white", linewidths=0.8, label=family, zorder=3,
        )

    share = float((gain > 0).mean())
    ax.set_xlabel("acceleration factor")
    ax.set_title(
        f"{gain_col.replace('psnr_gain_', '')}\n{share:.0%} of masks improved",
        fontsize=10,
    )
    ax.set_xticks(sorted(np.unique(x)))
    ax.set_xticklabels([f"{v:g}x" for v in sorted(np.unique(x))])
    ax.grid(True, alpha=0.18, linewidth=0.6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--y-column", default="psf_max_sidelobe")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run = run_dir(cfg)
    sweep_path = run / "metrics" / "budget_sweep.csv"
    if not sweep_path.exists():
        raise SystemExit(f"{sweep_path} not found; run scripts/11_budget_sweep.py first")
    table = pd.read_csv(sweep_path)

    all_gain_cols = [c for c in table.columns if c.startswith("psnr_gain_")]
    if not all_gain_cols:
        raise SystemExit("no psnr_gain_* columns in the sweep table")
    # A method whose gain never leaves the noise floor has nothing to show; an
    # all-white panel would only dilute the figure.
    gain_cols = [c for c in all_gain_cols if float(np.nanmax(np.abs(table[c]))) >= 0.05]
    for dropped in set(all_gain_cols) - set(gain_cols):
        print(f"omitting {dropped}: |gain| never exceeds 0.05 dB")

    # Symmetric about zero so the neutral midpoint lands exactly on the sign flip.
    vmax = float(np.nanmax(np.abs(table[gain_cols].to_numpy(dtype=float))))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    families = sorted(set(table["mask"].map(family_of)))

    fig, axes = plt.subplots(
        1, len(gain_cols), figsize=(5.6 * len(gain_cols), 5.6),
        squeeze=False, sharey=True,
    )
    for ax, gain_col in zip(axes[0], gain_cols):
        panel(ax, table, gain_col, args.y_column, norm, families)
    axes[0][0].set_ylabel("mask coherence (PSF max sidelobe)")

    mappable = plt.cm.ScalarMappable(norm=norm, cmap=DIVERGING)
    bar = fig.colorbar(mappable, ax=axes[0].tolist(), pad=0.02)
    bar.set_label("PSNR gain over zero-filling (dB)")
    bar.ax.axhline(0.0, color="#111111", linewidth=1.6)

    # One shared legend below the panels, plus a proxy for the zero contour, so
    # the boundary is named once instead of on every contour segment. Family
    # identity is carried by marker shape, so the proxies are neutral gray —
    # a colored swatch here would imply the color encodes family rather than gain.
    handles = [
        Line2D([], [], linestyle="none", marker=MARKERS[i % len(MARKERS)],
               markerfacecolor="#9aa0a6", markeredgecolor="white", markersize=8)
        for i in range(len(families))
    ]
    labels = list(families)
    handles.append(Line2D([], [], color="#111111", linewidth=2.0))
    labels.append("zero crossing (prior stops helping)")
    fig.legend(
        handles, labels, loc="lower center", ncol=min(6, len(labels)),
        fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.06),
    )
    fig.suptitle(
        "When does the prior help?  red = prior improves on zero-filling,  "
        "blue = prior degrades it",
        fontsize=11,
    )
    out = run / "plots" / "phase_diagram.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print(f"phase diagram: {out}")
    for gain_col in gain_cols:
        by_accel = table.groupby("acceleration")[gain_col].agg(["mean", "min", "max"])
        negative = (table[gain_col] < 0).groupby(table["acceleration"]).sum()
        print(f"\n{gain_col}:")
        print(by_accel.round(3).to_string())
        print("masks with negative gain per acceleration: " + negative.to_dict().__str__())


if __name__ == "__main__":
    main()
