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
