"""Artifact analysis: decomposition, artifact maps, PSF metrics, mask scores."""

from __future__ import annotations

from dataclasses import dataclass

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
    """Total error magnitude |recon - truth|.

    Backward-compatibility wrapper. This conflates observed-subspace error with
    null-space imputation error; use decompose_error for the structured split.
    """
    return (recon.to(torch.complex64) - truth.to(torch.complex64)).abs()


@dataclass
class ErrorDecomposition:
    """Reconstruction error split by the orthogonal projector P = F^H M F.

    consistency_error lives in the observed subspace (range of P); the
    artifact_field is the null-space part of the error, i.e. spurious or
    missing null-space content relative to the reference signal.
    """

    total_error: torch.Tensor        # |recon - truth|, magnitude-valued
    consistency_error: torch.Tensor  # P(recon - truth): complex, observed-subspace error
    artifact_field: torch.Tensor     # (I - P)(recon - truth): complex, null-space imputation error
    recon_nullspace: torch.Tensor    # (I - P) recon: complex, invented null-space content
    truth_nullspace: torch.Tensor    # (I - P) truth: complex, reference null-space component

    total_error_norm: float
    consistency_norm: float
    artifact_norm: float
    recon_nullspace_norm: float
    truth_nullspace_norm: float

    no_nullspace_content: bool


def decompose_error(
    recon: torch.Tensor,
    truth: torch.Tensor,
    mask: np.ndarray | torch.Tensor,
    *,
    rel_tol: float = 1e-4,
) -> ErrorDecomposition:
    """Decompose reconstruction error into observed-subspace and null-space parts.

    Single-image contract: recon and truth have shape (H, W); callers loop over
    batches. Inputs are cast to complex64 internally. All projections reuse the
    existing orthogonal projector; the null-space projection is computed as
    (I - P) z = z - projector(z, mask).

    no_nullspace_content is a norm-based flag: it is True whenever the
    reconstruction's null-space content is negligible relative to the reference
    signal norm. This holds for any reconstruction confined to the observed
    subspace (zero-filling and every linear diagonal method), not only
    zero-filling.
    """
    recon_c = recon.to(torch.complex64)
    truth_c = truth.to(torch.complex64)

    err = recon_c - truth_c
    consistency_error = projector(err, mask)
    artifact_field = err - consistency_error
    recon_nullspace = recon_c - projector(recon_c, mask)
    truth_nullspace = truth_c - projector(truth_c, mask)

    recon_nullspace_norm = torch.linalg.vector_norm(recon_nullspace).item()
    truth_norm = torch.linalg.vector_norm(truth_c).item()
    return ErrorDecomposition(
        total_error=err.abs(),
        consistency_error=consistency_error,
        artifact_field=artifact_field,
        recon_nullspace=recon_nullspace,
        truth_nullspace=truth_nullspace,
        total_error_norm=torch.linalg.vector_norm(err).item(),
        consistency_norm=torch.linalg.vector_norm(consistency_error).item(),
        artifact_norm=torch.linalg.vector_norm(artifact_field).item(),
        recon_nullspace_norm=recon_nullspace_norm,
        truth_nullspace_norm=torch.linalg.vector_norm(truth_nullspace).item(),
        no_nullspace_content=recon_nullspace_norm <= rel_tol * max(truth_norm, 1e-12),
    )


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
    """Predicted per-pixel zero-filled MSE from a mean frequency-domain power spectrum.

    With the orthonormal FFT, the zero-filled error energy equals the spectral
    energy at unsampled locations (Parseval), so the expected per-pixel MSE is
    the unsampled mean power divided by the number of pixels.
    """
    return float(((1.0 - mask) * mean_power).sum() / mask.size)
