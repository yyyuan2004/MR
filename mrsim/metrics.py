"""Scalar reconstruction quality metrics."""

from __future__ import annotations

import numpy as np
from skimage.metrics import structural_similarity

_EPS = 1e-12


def mse(recon: np.ndarray, truth: np.ndarray) -> float:
    return float(np.mean((np.asarray(recon, dtype=np.float64) - np.asarray(truth, dtype=np.float64)) ** 2))


def psnr(recon: np.ndarray, truth: np.ndarray, data_range: float = 1.0) -> float:
    return float(10.0 * np.log10(data_range**2 / max(mse(recon, truth), _EPS)))


def nrmse(recon: np.ndarray, truth: np.ndarray) -> float:
    truth = np.asarray(truth, dtype=np.float64)
    denom = max(float(np.linalg.norm(truth)), _EPS)
    return float(np.linalg.norm(np.asarray(recon, dtype=np.float64) - truth) / denom)


def ssim(recon: np.ndarray, truth: np.ndarray, data_range: float = 1.0) -> float:
    return float(
        structural_similarity(
            np.asarray(truth, dtype=np.float64),
            np.asarray(recon, dtype=np.float64),
            data_range=data_range,
        )
    )


def evaluate(recon: np.ndarray, truth: np.ndarray, data_range: float = 1.0) -> dict[str, float]:
    """All scalar metrics for one magnitude reconstruction against the truth."""
    return {
        "mse": mse(recon, truth),
        "psnr": psnr(recon, truth, data_range),
        "ssim": ssim(recon, truth, data_range),
        "nrmse": nrmse(recon, truth),
    }
