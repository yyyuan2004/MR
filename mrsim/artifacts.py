"""Artifact analysis: decomposition, artifact maps, PSF metrics, mask scores."""

from __future__ import annotations

import numpy as np
import torch

from .fft_ops import projector


def decompose(images: torch.Tensor, mask: np.ndarray | torch.Tensor) -> dict[str, torch.Tensor]:
    """Split images into the part visible to the mask and the part lost.

    P = F^H M F is an orthogonal projection, so kept = P x and
    missing = (I - P) x are orthogonal and satisfy
    ||x||^2 = ||kept||^2 + ||missing||^2.
    """
    x = images.to(torch.complex64)
    kept = projector(x, mask)
    return {"kept": kept, "missing": x - kept}


def artifact_map(recon: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    """Pointwise magnitude of the complex reconstruction error."""
    return (recon.to(torch.complex64) - truth.to(torch.complex64)).abs()


def aliasing_energy_ratio(image: torch.Tensor, mask: np.ndarray | torch.Tensor) -> float:
    """Fraction of image energy lost by the sampling projector, ||(I-P)x||^2 / ||x||^2."""
    parts = decompose(image, mask)
    total = float((parts["kept"].abs() ** 2).sum() + (parts["missing"].abs() ** 2).sum())
    if total == 0.0:
        return 0.0
    return float((parts["missing"].abs() ** 2).sum()) / total


def point_spread_function(mask: np.ndarray) -> np.ndarray:
    """Complex PSF of a mask, normalized to unit peak at the image center."""
    psf = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(mask), norm="ortho"))
    peak = psf[mask.shape[0] // 2, mask.shape[1] // 2]
    if np.abs(peak) == 0.0:
        raise ValueError("mask has no samples; PSF peak is zero")
    return psf / peak


def psf_metrics(mask: np.ndarray) -> dict[str, float]:
    """Sidelobe metrics of the mask PSF.

    max_sidelobe is the largest off-peak magnitude relative to the peak
    (the coherence of the mask). sidelobe_energy is the off-peak fraction of
    total PSF energy; by Parseval it depends mostly on the budget, not the
    arrangement, and is reported for completeness.
    """
    mag = np.abs(point_spread_function(mask))
    cy, cx = mask.shape[0] // 2, mask.shape[1] // 2
    peak = mag[cy, cx]
    side = mag.copy()
    side[cy, cx] = 0.0
    total_energy = float((mag**2).sum())
    return {
        "psf_max_sidelobe": float(side.max() / peak),
        "psf_sidelobe_energy": float((side**2).sum() / max(total_energy, 1e-12)),
    }


def expected_zero_filled_mse(mask: np.ndarray, mean_power: np.ndarray) -> float:
    """Predicted per-pixel zero-filled MSE from a mean k-space power spectrum.

    With the orthonormal FFT, the zero-filled error energy equals the spectral
    energy at unsampled locations (Parseval), so the expected per-pixel MSE is
    the unsampled mean power divided by the number of pixels.
    """
    return float(((1.0 - mask) * mean_power).sum() / mask.size)
