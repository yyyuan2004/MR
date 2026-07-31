"""Plot and image saving helpers (matplotlib, Agg backend)."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .artifacts import point_spread_function


def save_image(
    image: np.ndarray,
    path: Path,
    title: str | None = None,
    cmap: str = "gray",
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_image_grid(
    images: Sequence[np.ndarray],
    path: Path,
    titles: Sequence[str] | None = None,
    n_cols: int = 5,
    cmap: str = "gray",
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(images)
    n_cols = min(n_cols, n)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.2 * n_cols, 2.4 * n_rows), squeeze=False)
    for i, ax in enumerate(axes.ravel()):
        ax.set_axis_off()
        if i < n:
            ax.imshow(images[i], cmap=cmap, vmin=vmin, vmax=vmax)
            if titles is not None:
                ax.set_title(titles[i], fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_psf(mask: np.ndarray, path: Path, title: str | None = None) -> None:
    """Three-panel PSF plot: mask, log-magnitude PSF, center-row profile in dB."""
    path.parent.mkdir(parents=True, exist_ok=True)
    mag = np.abs(point_spread_function(mask))
    center_row = mask.shape[0] // 2
    profile_db = 20.0 * np.log10(mag[center_row, :] + 1e-12)

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
    axes[0].imshow(mask, cmap="gray")
    axes[0].set_title("mask")
    axes[0].set_axis_off()
    axes[1].imshow(20.0 * np.log10(mag + 1e-12), cmap="viridis", vmin=-60, vmax=0)
    axes[1].set_title("|PSF| (dB)")
    axes[1].set_axis_off()
    axes[2].plot(np.arange(mask.shape[1]) - mask.shape[1] // 2, profile_db)
    axes[2].set_title("center-row profile")
    axes[2].set_xlabel("pixel offset")
    axes[2].set_ylabel("dB")
    axes[2].set_ylim(-80, 5)
    axes[2].grid(True, alpha=0.3)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_series(
    values: Sequence[float],
    path: Path,
    xlabel: str = "step",
    ylabel: str = "value",
    title: str | None = None,
    logy: bool = False,
) -> None:
    """Plot a single scalar series (e.g. a trace history or training loss)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(np.arange(len(values)), values, linewidth=1.4)
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_psf_profiles(masks_dict: dict[str, np.ndarray], path: Path) -> None:
    """Overlay the center-row PSF profiles (dB) of several masks in one figure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, mask in masks_dict.items():
        mag = np.abs(point_spread_function(mask))
        row = mask.shape[0] // 2
        offsets = np.arange(mask.shape[1]) - mask.shape[1] // 2
        ax.plot(offsets, 20.0 * np.log10(mag[row, :] + 1e-12), label=name, linewidth=1.2)
    ax.set_xlabel("pixel offset")
    ax.set_ylabel("|PSF| (dB)")
    ax.set_ylim(-80, 5)
    ax.set_title("center-row PSF profiles")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _busiest_crop(truth: np.ndarray, crop: int) -> tuple[int, int]:
    """Top-left corner of the crop window with the largest local variance."""
    best, corner = -1.0, (0, 0)
    step = max(1, crop // 2)
    for y in range(0, truth.shape[0] - crop + 1, step):
        for x in range(0, truth.shape[1] - crop + 1, step):
            var = float(truth[y : y + crop, x : x + crop].std())
            if var > best:
                best, corner = var, (y, x)
    return corner


def plot_zoom_comparison(
    truth: np.ndarray,
    recons: dict[str, np.ndarray],
    path: Path,
    crop: int = 20,
    title: str | None = None,
) -> None:
    """Crop-and-zoom comparison: per mask, reconstruction, zoomed crop, zoomed error.

    The crop window is placed on the most structured region of the reference
    signal; the first row shows the reference itself.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    y0, x0 = _busiest_crop(truth, crop)
    rows = [("reference", truth)] + list(recons.items())
    err_max = max(float(np.abs(r - truth).max()) for _, r in recons.items())

    fig, axes = plt.subplots(len(rows), 3, figsize=(7.2, 2.3 * len(rows)), squeeze=False)
    for i, (name, image) in enumerate(rows):
        full, zoom, err = axes[i]
        full.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
        full.add_patch(
            plt.Rectangle((x0 - 0.5, y0 - 0.5), crop, crop, fill=False, edgecolor="red", lw=1.0)
        )
        full.set_ylabel(name, fontsize=8, rotation=0, ha="right", va="center")
        zoom.imshow(image[y0 : y0 + crop, x0 : x0 + crop], cmap="gray", vmin=0.0, vmax=1.0)
        err.imshow(
            np.abs(image - truth)[y0 : y0 + crop, x0 : x0 + crop],
            cmap="inferno",
            vmin=0.0,
            vmax=err_max,
        )
        for ax in (full, zoom, err):
            ax.set_xticks([])
            ax.set_yticks([])
        if i == 0:
            full.set_title("full", fontsize=9)
            zoom.set_title("zoom", fontsize=9)
            err.set_title("|error| zoom", fontsize=9)
    fig.tight_layout()
    if title:
        # Reserve headroom so the suptitle clears the first row's column titles.
        fig.subplots_adjust(top=1.0 - 0.45 / len(rows))
        fig.suptitle(title, fontsize=10)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_score_vs_error(
    masks_col: list[str],
    methods_col: list[str],
    scores: list[float],
    errors: list[float],
    path: Path,
) -> None:
    """Mask score vs measured error: one marker style per method, log-log axes,
    each mask labeled once (at its zero-filled point)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    markers = {"zero_filled": "o", "wiener": "s", "wavelet_ista": "^", "subspace": "D", "generative": "v"}
    for method in sorted(set(methods_col)):
        xs = [s for s, m in zip(scores, methods_col) if m == method]
        ys = [e for e, m in zip(errors, methods_col) if m == method]
        ax.scatter(xs, ys, s=42, marker=markers.get(method, "x"), label=method, alpha=0.8)
    for mask, method, s, e in zip(masks_col, methods_col, scores, errors):
        if method == "zero_filled":
            ax.annotate(mask, (s, e), textcoords="offset points", xytext=(6, 3), fontsize=7)
    lims = [min(min(scores), min(errors)), max(max(scores), max(errors))]
    ax.plot(lims, lims, linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("mask score (expected zero-filled complex MSE, train second moment)")
    ax.set_ylabel("measured mean complex MSE (test split)")
    ax.set_title("predicted mask score vs measured reconstruction error")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def representative_subset(
    masks_dict: dict[str, np.ndarray],
    train_power: np.ndarray,
    n: int = 8,
) -> dict[str, np.ndarray]:
    """Up to n masks spread across the observed-energy range.

    Overlay plots become unreadable past ~8 lines; picking quantiles of the
    observed training-energy fraction keeps both extremes and the middle of
    the family visible, deterministically.
    """
    if len(masks_dict) <= n:
        return dict(masks_dict)
    total = float(train_power.sum())
    ordered = sorted(
        masks_dict, key=lambda name: float((masks_dict[name] * train_power).sum()) / total
    )
    positions = np.unique(np.linspace(0, len(ordered) - 1, n).astype(int))
    return {ordered[p]: masks_dict[ordered[p]] for p in positions}


def plot_radial_coverage(
    masks_dict: dict[str, np.ndarray],
    train_power: np.ndarray,
    path: Path,
    reference_decay: float = 2.0,
    n_bins: int = 24,
) -> "pd.DataFrame":
    """Frequency-coverage diagnostic: radial density and radial rho profiles.

    Left panel: fraction of locations sampled per radius bin, with the
    repository's variable-density kernel at the reference decay (default 2,
    the classical optimal-density exponent) scaled to the same budget as a
    baseline. Right panel: the radial profile of rho — the fraction of the
    training spectral energy in each radius bin that the mask observes.
    Returns the underlying table so the profiles are testable.
    """
    import pandas as pd

    from .masks import radius_map

    path.parent.mkdir(parents=True, exist_ok=True)
    shape = train_power.shape
    r = radius_map(shape)
    edges = np.linspace(0.0, float(r.max()) + 1e-9, n_bins + 1)
    bin_of = np.clip(np.digitize(r.ravel(), edges) - 1, 0, n_bins - 1)
    counts = np.bincount(bin_of, minlength=n_bins).astype(np.float64)
    # A radius bin with no grid locations has no density or rho to report;
    # NaN makes matplotlib skip it instead of plotting a silent zero.
    counts_safe = np.where(counts > 0, counts, np.nan)
    power_per_bin = np.bincount(bin_of, weights=train_power.ravel(), minlength=n_bins)
    centers = 0.5 * (edges[:-1] + edges[1:])

    budgets = {name: float(mask.sum()) for name, mask in masks_dict.items()}
    budget = float(np.mean(list(budgets.values())))
    # Same kernel family the variable-density generator uses, at the reference
    # decay, scaled so its expected sample count matches the budget.
    kernel = (1.0 + r / (0.05 * max(shape))) ** (-reference_decay)
    kernel = np.minimum(1.0, kernel * budget / kernel.sum())
    reference = np.bincount(bin_of, weights=kernel.ravel(), minlength=n_bins) / counts_safe

    rows = []
    fig, (ax_density, ax_rho) = plt.subplots(1, 2, figsize=(11.5, 4.4))
    for name, mask in masks_dict.items():
        sampled = np.bincount(bin_of, weights=mask.ravel(), minlength=n_bins)
        observed = np.bincount(
            bin_of, weights=(mask * train_power).ravel(), minlength=n_bins
        )
        density = sampled / counts_safe
        rho_profile = np.where(
            power_per_bin > 0, observed / np.maximum(power_per_bin, 1e-30), np.nan
        )
        ax_density.plot(centers, density, linewidth=1.3, label=name)
        ax_rho.plot(centers, rho_profile, linewidth=1.3, label=name)
        for b in range(n_bins):
            rows.append(
                {
                    "mask": name,
                    "radius": float(centers[b]),
                    "sampled": float(sampled[b]),
                    "density": float(density[b]),
                    "rho_profile": float(rho_profile[b]),
                }
            )
    ax_density.plot(
        centers, reference, linewidth=1.6, linestyle="--", color="#555555",
        label=f"decay={reference_decay:g} reference",
    )
    ax_density.set_xlabel("radius (frequency-domain)")
    ax_density.set_ylabel("fraction of locations sampled")
    ax_density.set_title("radial sampling density")
    ax_rho.set_xlabel("radius (frequency-domain)")
    ax_rho.set_ylabel("observed energy fraction")
    ax_rho.set_title("radial profile of rho")
    ax_rho.set_ylim(-0.02, 1.02)
    for ax in (ax_density, ax_rho):
        ax.grid(True, alpha=0.25)
    ax_density.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(rows)


def plot_subband_leakage(
    masks_dict: dict[str, np.ndarray],
    mass: dict[str, np.ndarray],
    energies: dict[str, float],
    path: Path,
) -> "pd.DataFrame":
    """Wavelet-energy-coverage diagnostic: per-subband leakage heatmap.

    Cell (mask, subband) is the fraction of that subband's spectral mass on
    unmeasured locations; the energy-weighted row sum reproduces
    artifacts.wavelet_leakage_score exactly, which is asserted in tests.
    Rows are sorted by total weighted leakage. Returns the leakage table.
    """
    import pandas as pd

    path.parent.mkdir(parents=True, exist_ok=True)
    bands = list(mass)
    total_energy = sum(energies.values())
    weights = np.array([energies[b] / total_energy for b in bands])

    names = list(masks_dict)
    leak = np.array(
        [
            [float((mass[b] * (1.0 - masks_dict[name])).sum()) for b in bands]
            for name in names
        ]
    )
    order = np.argsort(leak @ weights)
    leak = leak[order]
    names = [names[i] for i in order]

    fig, ax = plt.subplots(figsize=(1.1 + 0.65 * len(bands), 1.2 + 0.32 * len(names)))
    image = ax.imshow(leak, cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(bands)))
    ax.set_xticklabels(
        [f"{b}\n(w={w:.2f})" for b, w in zip(bands, weights)], fontsize=7
    )
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=7)
    ax.set_title("per-subband null-space leakage (energy weight w in labels)", fontsize=9)
    fig.colorbar(image, ax=ax, label="leakage fraction")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    table = pd.DataFrame(leak, columns=bands)
    table.insert(0, "mask", names)
    table["weighted_total"] = leak @ weights
    return table


def plot_singular_spectra(
    spectra: dict[str, np.ndarray],
    path: Path,
    floor: float = 1e-12,
) -> None:
    """Full singular spectrum of the restricted operator, per mask (log scale).

    sigma_min alone hides rank deficiency: every singular value at the
    numerical floor is an unrecoverable direction, and ranking masks by a
    noise-level minimum is meaningless. The full curve shows where each
    spectrum crosses the floor and how many directions survive.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for name, values in spectra.items():
        values = np.sort(np.asarray(values, dtype=np.float64))[::-1]
        ax.plot(
            np.arange(1, values.size + 1),
            np.maximum(values, floor),
            linewidth=1.3,
            label=f"{name} ({int((values > floor).sum())}/{values.size} above floor)",
        )
    ax.axhline(floor, color="#555555", linestyle="--", linewidth=1.0)
    ax.set_yscale("log")
    ax.set_xlabel("singular value index")
    ax.set_ylabel("singular value")
    ax.set_title("restricted-operator singular spectra (dashed: numerical floor)")
    ax.grid(True, which="both", alpha=0.2)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_subband_sigma_min(
    profiles: dict[str, dict[str, float]],
    path: Path,
    floor: float = 1e-12,
) -> None:
    """Per-subband smallest singular value of the restricted operator.

    2-D wavelet orientation bands (horizontal/vertical/diagonal) encode
    direction and the level hierarchy encodes scale, so this profile shows
    *which* directions and scales a mask leaves ill-conditioned — e.g. a
    Cartesian column mask collapses the bands whose spectral mass falls
    between the sampled lines while leaving the orthogonal orientation
    conditioned. Rank-deficient bands (sigma_min = 0) are drawn at the floor.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    bands = list(next(iter(profiles.values())))
    x = np.arange(len(bands))
    fig, ax = plt.subplots(figsize=(1.6 + 0.9 * len(bands), 4.6))
    for name, profile in profiles.items():
        values = np.array([profile.get(band, np.nan) for band in bands], dtype=np.float64)
        ax.plot(x, np.maximum(values, floor), marker="o", markersize=4,
                linewidth=1.3, label=name)
    ax.axhline(floor, color="#555555", linestyle="--", linewidth=1.0)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(bands, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("sigma_min of the restricted operator")
    ax.set_title("per-subband conditioning (level 1 = coarsest; dashed: floor)")
    ax.grid(True, which="both", alpha=0.2)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# Compressed display names for parameterized family members. Prefix mapping
# plus suffix compression, so "variable_density_d2.5_s0" -> "vd2.5s0" without
# per-mask entries.
_SHORT_PREFIXES = [
    ("variable_density_lines", "vdl"),
    ("variable_density", "vd"),
    ("uniform_random", "uni"),
    ("multilevel", "ml"),
    ("psf_penalized", "psf"),
    ("subspace_aopt", "sub"),
    ("aopt_greedy", "aopt"),
    ("equispaced_lines", "equi"),
    ("line_subspace_leakage", "l-leak"),
    ("line_aopt", "l-aopt"),
    ("spectrum_energy_greedy", "spec-e"),
    ("recon_in_loop_greedy", "ril"),
    ("data_driven_greedy", "dd"),
    ("loupe_learned", "loupe"),
]

# Fixed family order (never cycled): color and marker follow the family.
_FAMILY_ORDER = ["uniform", "vd", "multilevel", "lines", "psf", "subspace", "other"]
_FAMILY_MARKERS = {"uniform": "o", "vd": "^", "multilevel": "D", "lines": "s",
                   "psf": "v", "subspace": "X", "other": "P"}


def short_name(name: str) -> str:
    """Compress a mask name for on-plot labels (vd2.5s0, psf-b4, mlL4d1.5)."""
    import re

    for prefix, code in _SHORT_PREFIXES:
        if name == prefix:
            return code
        if name.startswith(prefix + "_"):
            parts = name[len(prefix) + 1 :].split("_")
            # A leading decay tag is redundant once the family code names it
            # (vd2.5s0); elsewhere the letter must stay or numbers merge
            # ambiguously (ml-L4d1.5, not ml-L41.5).
            if parts[0][:1] == "d" and re.fullmatch(r"d\d+(?:\.\d+)?", parts[0]):
                parts[0] = parts[0][1:]
            compressed = "".join(parts)
            separator = "" if compressed[:1].isdigit() else "-"
            return f"{code}{separator}{compressed}"
    return name


def design_family(name: str) -> str:
    """Design family of a mask name, for color/marker assignment."""
    if name.startswith("uniform_random"):
        return "uniform"
    if name.startswith("variable_density_lines"):
        return "lines"
    if name.startswith("variable_density"):
        return "vd"
    if name.startswith("multilevel"):
        return "multilevel"
    if name.startswith(("equispaced", "line_", "spectrum_energy", "loupe")):
        return "lines"
    # aopt_greedy is the beta = 0 member of the PSF-penalized family.
    if name.startswith(("psf_penalized", "aopt_greedy")):
        return "psf"
    if name.startswith("subspace_aopt"):
        return "subspace"
    return "other"


def scatter_with_labels(
    x: Sequence[float],
    y: Sequence[float],
    labels: Sequence[str],
    path: Path,
    xlabel: str,
    ylabel: str,
    title: str | None = None,
    max_direct_labels: int = 6,
    always_label: Sequence[str] = (),
) -> None:
    """Family-colored scatter with selective direct labels.

    Color and marker follow the design family (fixed order, with a legend);
    direct labels use compressed names and are attached only to the extreme
    points — the bottom/top ``max_direct_labels // 2`` by y — plus any names
    in ``always_label``. Labeling all points is unreadable past ~10 masks.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    families = [design_family(label) for label in labels]
    colors = plt.get_cmap("tab10")

    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    for family in _FAMILY_ORDER:
        idx = [i for i, f in enumerate(families) if f == family]
        if not idx:
            continue
        ax.scatter(
            x[idx], y[idx], s=46,
            color=colors(_FAMILY_ORDER.index(family)),
            marker=_FAMILY_MARKERS[family],
            edgecolors="white", linewidths=0.6, label=family, alpha=0.9,
        )

    half = max(1, max_direct_labels // 2)
    order = np.argsort(y)
    to_label = set(order[:half]) | set(order[-half:])
    to_label |= {i for i, label in enumerate(labels) if label in set(always_label)}
    for i in to_label:
        ax.annotate(
            short_name(labels[i]), (x[i], y[i]),
            textcoords="offset points", xytext=(6, 4), fontsize=7.5,
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7.5, title="family", title_fontsize=7.5)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
