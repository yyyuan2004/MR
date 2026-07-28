"""Scalar reconstruction quality metrics."""

from __future__ import annotations

import numpy as np
from skimage.metrics import structural_similarity

_EPS = 1e-12


def mse(recon: np.ndarray, truth: np.ndarray) -> float:
    """Legacy value-domain MSE for explicitly real-valued arrays.

    Complex inputs are rejected rather than silently discarding phase or the
    imaginary component.  Use :func:`complex_mse` or :func:`magnitude_mse` to
    state the intended complex-image semantics.
    """
    recon_array = np.asarray(recon)
    truth_array = np.asarray(truth)
    if np.iscomplexobj(recon_array) or np.iscomplexobj(truth_array):
        raise ValueError("mse is real-valued; use complex_mse or magnitude_mse")
    difference = recon_array.astype(np.float64) - truth_array.astype(np.float64)
    return float(np.mean(difference**2))


def complex_mse(recon: np.ndarray, truth: np.ndarray) -> float:
    """Mean squared complex error ``mean(|recon - truth|^2)``."""
    difference = np.asarray(recon, dtype=np.complex128) - np.asarray(
        truth, dtype=np.complex128
    )
    return float(np.mean(np.abs(difference) ** 2))


def magnitude_mse(recon: np.ndarray, truth: np.ndarray) -> float:
    """MSE between image magnitudes ``mean((|recon| - |truth|)^2)``."""
    difference = np.abs(np.asarray(recon)) - np.abs(np.asarray(truth))
    return float(np.mean(np.asarray(difference, dtype=np.float64) ** 2))


def psnr(recon: np.ndarray, truth: np.ndarray, data_range: float = 1.0) -> float:
    return float(10.0 * np.log10(data_range**2 / max(mse(recon, truth), _EPS)))


def nrmse(recon: np.ndarray, truth: np.ndarray) -> float:
    truth = np.asarray(truth, dtype=np.float64)
    denom = max(float(np.linalg.norm(truth)), _EPS)
    return float(np.linalg.norm(np.asarray(recon, dtype=np.float64) - truth) / denom)


def ssim(recon: np.ndarray, truth: np.ndarray, data_range: float = 1.0) -> float:
    truth_array = np.asarray(truth, dtype=np.float64)
    recon_array = np.asarray(recon, dtype=np.float64)
    min_side = min(truth_array.shape[-2:])
    win_size = min(7, min_side)
    if win_size % 2 == 0:
        win_size -= 1
    if win_size < 3:
        raise ValueError("SSIM requires both image dimensions to be at least 3")
    return float(
        structural_similarity(
            truth_array,
            recon_array,
            data_range=data_range,
            win_size=win_size,
        )
    )


def evaluate(recon: np.ndarray, truth: np.ndarray, data_range: float = 1.0) -> dict[str, float]:
    """All scalar metrics for one magnitude reconstruction against the truth.

    Complex inputs are converted to magnitudes explicitly.  The historical
    ``"mse"`` output key therefore denotes :func:`magnitude_mse`, not complex
    MSE.
    """
    recon_magnitude = np.abs(np.asarray(recon))
    truth_magnitude = np.abs(np.asarray(truth))
    return {
        "mse": magnitude_mse(recon, truth),
        "psnr": psnr(recon_magnitude, truth_magnitude, data_range),
        "ssim": ssim(recon_magnitude, truth_magnitude, data_range),
        "nrmse": nrmse(recon_magnitude, truth_magnitude),
    }
