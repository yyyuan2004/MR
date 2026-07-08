"""Artifact analysis: decomposition, artifact maps, PSF metrics, mask scores."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pywt
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


def spectrum_weighted_psf_metrics(
    mask: np.ndarray, spectrum: np.ndarray, guard_radius: float = 3.0
) -> dict[str, float]:
    """Sidelobe metrics of the prior-weighted PSF.

    The plain PSF treats every frequency equally, so its metrics cannot by
    themselves rank masks whose difference lies in *where* spectral energy
    sits. Weighting the mask by a prior power spectrum before the transform
    measures coherent aliasing of the expected signal.

    The weighted PSF is dominated by the prior's autocorrelation main lobe
    (steep priors concentrate energy at the frequency-domain center), so lags
    within guard_radius of the peak are excluded; without the guard the
    metric saturates near 1 for every mask and cannot rank them.
    """
    weighted = np.fft.fftshift(
        np.fft.ifft2(np.fft.ifftshift(mask * spectrum), norm="ortho")
    )
    mag = np.abs(weighted)
    cy, cx = mask.shape[0] // 2, mask.shape[1] // 2
    peak = float(mag[cy, cx])
    if peak == 0.0:
        raise ValueError("mask/spectrum product is zero; weighted PSF peak is zero")
    yy, xx = np.meshgrid(
        np.arange(mask.shape[0]) - cy, np.arange(mask.shape[1]) - cx, indexing="ij"
    )
    side = mag.copy()
    side[np.hypot(yy, xx) <= guard_radius] = 0.0
    return {"weighted_max_sidelobe": float(side.max() / peak)}


def subband_spectral_mass(
    shape: tuple[int, int], wavelet: str = "db4", levels: int = 3
) -> dict[str, np.ndarray]:
    """Normalized frequency-domain energy distribution of each wavelet subband.

    Atoms within one subband are translates of each other, so they share one
    magnitude spectrum; a single centered atom per subband suffices. Each
    returned array sums to 1 over the frequency grid.
    """
    template = pywt.wavedec2(
        np.zeros(shape), wavelet=wavelet, mode="periodization", level=levels
    )
    mass: dict[str, np.ndarray] = {}

    def atom_mass(coeffs: list) -> np.ndarray:
        atom = pywt.waverec2(coeffs, wavelet=wavelet, mode="periodization")
        power = np.abs(np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(atom), norm="ortho"))) ** 2
        return power / power.sum()

    approx = [np.zeros_like(template[0])] + [
        tuple(np.zeros_like(d) for d in band) for band in template[1:]
    ]
    cy, cx = template[0].shape[0] // 2, template[0].shape[1] // 2
    approx[0][cy, cx] = 1.0
    mass["approx"] = atom_mass(approx)
    approx[0][cy, cx] = 0.0

    for level_index, band in enumerate(template[1:], start=1):
        for detail_index, name in enumerate(("horizontal", "vertical", "diagonal")):
            coeffs = [np.zeros_like(template[0])] + [
                tuple(np.zeros_like(d) for d in b) for b in template[1:]
            ]
            target = coeffs[level_index][detail_index]
            target[target.shape[0] // 2, target.shape[1] // 2] = 1.0
            mass[f"level{level_index}_{name}"] = atom_mass(coeffs)
    return mass


def subband_energies(
    images: np.ndarray, wavelet: str = "db4", levels: int = 3
) -> dict[str, float]:
    """Total wavelet-coefficient energy per subband over a stack of images."""
    coeffs = pywt.wavedec2(
        np.asarray(images, dtype=np.float64),
        wavelet=wavelet,
        mode="periodization",
        level=levels,
        axes=(-2, -1),
    )
    energies = {"approx": float((np.abs(coeffs[0]) ** 2).sum())}
    for level_index, band in enumerate(coeffs[1:], start=1):
        for detail, name in zip(band, ("horizontal", "vertical", "diagonal")):
            energies[f"level{level_index}_{name}"] = float((np.abs(detail) ** 2).sum())
    return energies


def wavelet_leakage_score(
    mask: np.ndarray,
    mass: dict[str, np.ndarray],
    energies: dict[str, float],
) -> float:
    """Energy-weighted null-space leakage of the wavelet basis under a mask.

    For each subband, the fraction of its atoms' spectral mass that falls on
    unmeasured locations is the share of that subband's content lost to the
    null space (a diagonal surrogate: cross-atom interference is ignored).
    Weighting by training-data subband energies gives the expected null-space
    energy of the reconstruction basis — an information-coverage score that,
    unlike plain PSF coherence, accounts for where signal energy actually is.
    Lower is better.
    """
    unmeasured = 1.0 - mask
    total_energy = sum(energies.values())
    leak = sum(energies[key] * float((mass[key] * unmeasured).sum()) for key in mass)
    return leak / max(total_energy, 1e-12)


def subspace_nullspace_leakage(
    basis: np.ndarray,
    mask: np.ndarray,
    weights: np.ndarray | None = None,
) -> float:
    """Fraction of subspace-basis energy lost to the null space under a mask.

    For Phi = F B, column j loses sum_{k unmeasured} |Phi_kj|^2 of its energy
    to the null space. The columns are combined weighted by their energy
    (optionally rescaled by `weights`, e.g. singular values), which reduces to
    ||(I - P) B W||_F^2 / ||B W||_F^2 — the subspace analogue of
    aliasing_energy_ratio. Lower means the mask observes more of the prior
    model; 0 means the subspace is fully measured.
    """
    from .subspace import to_kspace_basis

    phi = to_kspace_basis(np.asarray(basis), mask.shape)
    power = np.abs(phi) ** 2
    scale = np.ones(phi.shape[1]) if weights is None else np.asarray(weights, dtype=np.float64) ** 2
    unmeasured = (1.0 - mask.ravel())[:, None]
    lost = float((power * unmeasured).sum(axis=0) @ scale)
    total = float(power.sum(axis=0) @ scale)
    return lost / max(total, 1e-30)


def expected_zero_filled_mse(mask: np.ndarray, mean_power: np.ndarray) -> float:
    """Predicted per-pixel zero-filled MSE from a mean frequency-domain power spectrum.

    With the orthonormal FFT, the zero-filled error energy equals the spectral
    energy at unsampled locations (Parseval), so the expected per-pixel MSE is
    the unsampled mean power divided by the number of pixels.
    """
    return float(((1.0 - mask) * mean_power).sum() / mask.size)
