"""Shared experiment plumbing used by the numbered scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch

from . import artifacts, data, greedy, masks, metrics, recon, viz
from .fft_ops import fft2c


# ---------------------------------------------------------------------------
# Dataset handling
# ---------------------------------------------------------------------------

def dataset_path(run: Path) -> Path:
    return run / "data" / "dataset.pt"


def load_or_generate_dataset(cfg: dict[str, Any], run: Path, force: bool = False) -> torch.Tensor:
    """Load the run's dataset, generating and saving it if needed."""
    path = dataset_path(run)
    if path.exists() and not force:
        return torch.load(path)
    d = cfg["data"]
    images = data.generate_dataset(
        n_images=int(d["n_images"]),
        size=int(d["image_size"]),
        seed=int(cfg["seed"]),
        phantom=str(d.get("phantom", "ellipses")),
        min_ellipses=int(d.get("min_ellipses", 3)),
        max_ellipses=int(d.get("max_ellipses", 8)),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(images, path)
    return images


def train_test_split(images: torch.Tensor, cfg: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic split: first n_train images train, next n_test images test."""
    d = cfg["data"]
    n_train, n_test = int(d["n_train"]), int(d["n_test"])
    if n_train + n_test > images.shape[0]:
        raise ValueError("n_train + n_test exceeds the dataset size")
    return images[:n_train], images[n_train : n_train + n_test]


# ---------------------------------------------------------------------------
# Spectra and mask construction
# ---------------------------------------------------------------------------

def mean_power_spectrum(images: torch.Tensor) -> np.ndarray:
    """Mean centered k-space power |X_k|^2 over a stack of images."""
    k = fft2c(images.to(torch.complex64))
    return (k.abs() ** 2).mean(dim=0).numpy().astype(np.float64)


def fitted_power_law_spectrum(images: torch.Tensor) -> np.ndarray:
    """Radially symmetric power-law fit to the empirical mean spectrum.

    Fits log s = intercept + slope * log(1 + r) by least squares and rebuilds
    a smooth spectrum from the radius map. Used as the model prior for
    A-optimal and artifact-aware greedy selection.
    """
    empirical = mean_power_spectrum(images)
    r = masks.radius_map(empirical.shape)
    x = np.log1p(r).ravel()
    y = np.log(empirical.ravel() + 1e-12)
    slope, intercept = np.polyfit(x, y, 1)
    spectrum = np.exp(intercept) * (1.0 + r) ** slope
    return np.maximum(spectrum, 1e-12)


def mask_budgets(cfg: dict[str, Any]) -> tuple[tuple[int, int], int, int]:
    """Return (shape, n_samples, n_center) from the config."""
    size = int(cfg["data"]["image_size"])
    shape = (size, size)
    n_samples = masks.budget_from_fraction(shape, float(cfg["mask"]["sampling_fraction"]))
    n_center = int(round(float(cfg["mask"].get("center_fraction", 0.0)) * n_samples))
    return shape, n_samples, min(n_center, n_samples)


def build_masks(
    names: list[str],
    cfg: dict[str, Any],
    train_images: torch.Tensor,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Build the requested masks by name, sharing one budget and prior."""
    shape, n_samples, n_center = mask_budgets(cfg)
    g = cfg.get("greedy", {})
    noise_var = float(g.get("noise_var", 1e-3))

    prior: np.ndarray | None = None
    builders: dict[str, Callable[[], np.ndarray]] = {
        "uniform_random": lambda: masks.uniform_random_mask(shape, n_samples, rng, n_center),
        "variable_density": lambda: masks.variable_density_mask(
            shape, n_samples, rng,
            decay=float(cfg["mask"].get("variable_density_decay", 3.0)),
            n_center=n_center,
        ),
        "equispaced_lines": lambda: masks.equispaced_lines_mask(shape, n_samples),
        "aopt_greedy": lambda: greedy.greedy_a_optimal(
            _prior(), n_samples, noise_var=noise_var, n_center=n_center
        ),
        "artifact_aware_greedy": lambda: greedy.greedy_artifact_aware(
            _prior(), n_samples,
            noise_var=noise_var,
            beta=float(g.get("artifact_beta", 0.5)),
            n_candidates=int(g.get("n_candidates", 32)),
            n_center=n_center,
        ),
        "data_driven_greedy": lambda: greedy.greedy_data_driven(
            train_images.numpy(), n_samples, n_center=n_center
        ),
    }

    def _prior() -> np.ndarray:
        nonlocal prior
        if prior is None:
            prior = fitted_power_law_spectrum(train_images)
        return prior

    out: dict[str, np.ndarray] = {}
    for name in names:
        if name not in builders:
            raise ValueError(f"unknown mask type {name!r}; known: {sorted(builders)}")
        out[name] = builders[name]()
    return out


def save_mask_bundle(mask: np.ndarray, name: str, run: Path) -> dict[str, float]:
    """Save mask array, mask image, and PSF plot; return PSF metrics row."""
    _ensure_dir(run / "masks")
    np.save(run / "masks" / f"{name}.npy", mask)
    viz.save_image(mask, run / "masks" / f"{name}.png", title=name)
    viz.plot_psf(mask, run / "psf" / f"{name}_psf.png", title=name)
    row: dict[str, float] = {"mask": name, "n_samples": float(mask.sum())}
    row.update(artifacts.psf_metrics(mask))
    return row


def _ensure_dir(path: Path) -> bool:
    path.mkdir(parents=True, exist_ok=True)
    return True


# ---------------------------------------------------------------------------
# Reconstruction and evaluation
# ---------------------------------------------------------------------------

def reconstruct_all(
    images: torch.Tensor,
    mask: np.ndarray,
    cfg: dict[str, Any],
    generator: torch.Generator | None = None,
) -> dict[str, torch.Tensor]:
    """Simulate measurements and reconstruct with every configured method."""
    noise_std = float(cfg.get("measurement", {}).get("noise_std", 0.0))
    lam = float(cfg["recon"]["ridge_lambda"])
    y = recon.simulate_measurements(images, mask, noise_std=noise_std, generator=generator)
    return {
        "zero_filled": recon.zero_filled(y),
        "ridge": recon.ridge(y, mask, lam),
    }


def metrics_rows(
    recons: dict[str, torch.Tensor],
    truth: torch.Tensor,
    mask: np.ndarray,
    mask_name: str,
) -> list[dict[str, Any]]:
    """Per-image metric rows for every method, including artifact metrics."""
    rows = []
    for method, batch in recons.items():
        for i in range(truth.shape[0]):
            truth_i = truth[i].numpy()
            recon_i = batch[i].abs().numpy()
            row: dict[str, Any] = {"mask": mask_name, "method": method, "image_index": i}
            row.update(metrics.evaluate(recon_i, truth_i))
            row["aliasing_energy_ratio"] = artifacts.aliasing_energy_ratio(truth[i], mask)
            rows.append(row)
    return rows


def save_examples(
    truth: torch.Tensor,
    recons: dict[str, torch.Tensor],
    mask_name: str,
    run: Path,
    indices: list[int],
) -> None:
    """Save reconstruction and artifact-map grids for selected test images."""
    for method, batch in recons.items():
        recon_images = [batch[i].abs().numpy() for i in indices]
        error_maps = [artifacts.artifact_map(batch[i], truth[i]).numpy() for i in indices]
        titles = [f"test[{i}]" for i in indices]
        viz.save_image_grid(
            recon_images,
            run / "recon" / f"{mask_name}_{method}.png",
            titles=titles,
            vmin=0.0,
            vmax=1.0,
        )
        viz.save_image_grid(
            error_maps,
            run / "artifact_maps" / f"{mask_name}_{method}.png",
            titles=titles,
            cmap="inferno",
        )


def evaluate_masks(
    mask_dict: dict[str, np.ndarray],
    test_images: torch.Tensor,
    cfg: dict[str, Any],
    run: Path,
    prefix: str,
    example_indices: list[int] | None = None,
) -> pd.DataFrame:
    """Full evaluation of a set of masks: metrics CSV, examples, PSF metrics.

    Returns the per-image metrics DataFrame (also written to
    runs/<exp>/metrics/<prefix>_metrics.csv).
    """
    generator = torch.Generator().manual_seed(int(cfg["seed"]))
    n_examples = int(cfg.get("outputs", {}).get("n_examples", 5))
    if example_indices is None:
        # Evenly spaced test indices as representative examples.
        example_indices = np.unique(
            np.linspace(0, test_images.shape[0] - 1, n_examples).astype(int)
        ).tolist()

    all_rows: list[dict[str, Any]] = []
    psf_rows: list[dict[str, float]] = []
    for name, mask in mask_dict.items():
        psf_rows.append(save_mask_bundle(mask, name, run))
        recons = reconstruct_all(test_images, mask, cfg, generator=generator)
        all_rows.extend(metrics_rows(recons, test_images, mask, name))
        save_examples(test_images, recons, name, run, example_indices)

    _ensure_dir(run / "metrics")
    frame = pd.DataFrame(all_rows)
    frame.to_csv(run / "metrics" / f"{prefix}_metrics.csv", index=False)
    pd.DataFrame(psf_rows).to_csv(run / "metrics" / f"{prefix}_psf_metrics.csv", index=False)
    return frame
