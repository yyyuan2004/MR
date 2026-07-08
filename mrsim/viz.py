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
    markers = {"zero_filled": "o", "wiener": "s", "wavelet_ista": "^"}
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
    ax.set_xlabel("mask score (expected zero-filled MSE, train spectrum)")
    ax.set_ylabel("measured mean MSE (test split)")
    ax.set_title("predicted mask score vs measured reconstruction error")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def scatter_with_labels(
    x: Sequence[float],
    y: Sequence[float],
    labels: Sequence[str],
    path: Path,
    xlabel: str,
    ylabel: str,
    title: str | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(x, y, s=40)
    for xi, yi, label in zip(x, y, labels):
        ax.annotate(label, (xi, yi), textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
