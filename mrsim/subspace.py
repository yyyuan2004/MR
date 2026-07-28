"""Subspace and manifold priors for sampling design and reconstruction.

The prior is abstracted as a pluggable N x d basis matrix B whose columns
span the model of plausible signals. A linear subspace uses the SVD basis
(B = U); a generator manifold uses the Jacobian of a fixed pre-trained
generator at a reference latent point (B = J_G(z0)). Both follow the same
selection and reconstruction code path. NumPy throughout; PyTorch is used
only at the generator boundary.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .fft_ops import fft2c


@dataclass(frozen=True)
class SubspaceStatistics:
    """SVD subspace together with the statistics needed by a Gaussian prior.

    ``eigenvalues`` contains the coefficient variances of the retained modes.
    For a centered fit these are the leading sample-covariance eigenvalues;
    for a legacy through-origin fit they are leading second-moment
    eigenvalues.  ``mean`` is always the empirical image mean as a flat vector.
    """

    basis: np.ndarray
    mean: np.ndarray
    eigenvalues: np.ndarray
    energy_ratio: float
    total_variance: float
    image_shape: tuple[int, int]
    centered: bool

    @property
    def mean_image(self) -> np.ndarray:
        """Empirical mean reshaped to the original image dimensions."""
        return self.mean.reshape(self.image_shape)

    @property
    def covariance_factor(self) -> np.ndarray:
        """Matrix ``B sqrt(Lambda)`` whose product forms the fitted covariance."""
        return self.basis * np.sqrt(self.eigenvalues)[None, :]


def fit_subspace(
    train_images: np.ndarray | torch.Tensor,
    d: int,
    *,
    center: bool = False,
    return_statistics: bool = False,
) -> tuple[np.ndarray, float] | SubspaceStatistics:
    """Fit a d-dimensional linear subspace to vectorized training images.

    Existing two-argument calls retain the historical through-origin fit and
    return ``(basis, energy_ratio)``.  For a statistically specified prior,
    call ``fit_subspace(..., center=True, return_statistics=True)`` to obtain
    the empirical mean, covariance eigenvalues, basis, and explained-variance
    ratio in a :class:`SubspaceStatistics` object.
    """
    if isinstance(train_images, torch.Tensor):
        train_images = train_images.detach().cpu().numpy()
    raw_images = np.asarray(train_images)
    dtype = np.complex128 if np.iscomplexobj(raw_images) else np.float64
    images = np.asarray(raw_images, dtype=dtype)
    if images.ndim != 3:
        raise ValueError("train_images must have shape (n, H, W)")
    n = images.shape[0]
    data = images.reshape(n, -1)
    max_rank = min(n - 1, data.shape[1]) if center else min(n, data.shape[1])
    if not (1 <= d <= max_rank):
        qualifier = " for a centered fit" if center else ""
        raise ValueError(f"d={d} must be in [1, {max_rank}]{qualifier}")

    mean = data.mean(axis=0)
    fitted_data = data - mean if center else data
    _, singular_values, vt = np.linalg.svd(fitted_data, full_matrices=False)
    basis = vt[:d].conj().T
    energy = singular_values**2
    total_energy = float(energy.sum())
    energy_ratio = float(energy[:d].sum() / total_energy) if total_energy > 0.0 else 0.0
    normalization = n - 1 if center else n
    eigenvalues = np.asarray(energy[:d] / normalization, dtype=np.float64)
    statistics = SubspaceStatistics(
        basis=basis,
        mean=mean,
        eigenvalues=eigenvalues,
        energy_ratio=energy_ratio,
        total_variance=total_energy / normalization,
        image_shape=(images.shape[1], images.shape[2]),
        centered=center,
    )
    if return_statistics:
        return statistics
    return statistics.basis, statistics.energy_ratio


def to_kspace_basis(basis: np.ndarray, shape: tuple[int, int] | None = None) -> np.ndarray:
    """Frequency-domain basis Phi = F B, applying fft2c column-wise.

    Accepts any N x d (real or complex) basis; returns complex128 N x d.
    """
    basis = np.asarray(basis)
    n_pixels, d = basis.shape
    if shape is None:
        side = int(round(np.sqrt(n_pixels)))
        if side * side != n_pixels:
            raise ValueError("cannot infer a square shape; pass shape explicitly")
        shape = (side, side)
    columns = basis.T.reshape(d, *shape).astype(np.complex128)
    phi = fft2c(torch.from_numpy(columns)).numpy()
    return phi.reshape(d, n_pixels).T


def generator_jacobian_basis(
    generator, z0: torch.Tensor, orthonormalize: bool = True
) -> np.ndarray:
    """Tangent basis of a generator manifold: B = J_G(z0), reshaped to N x d.

    The generator maps a latent vector (d,) to an image (H, W). The Jacobian
    at the reference point z0 spans the local tangent space of the generator
    manifold; QR orthonormalization (default) makes the columns directly
    comparable to an SVD subspace basis and stabilizes the selection math.
    """
    jac = torch.autograd.functional.jacobian(generator, z0.detach())
    basis = jac.reshape(-1, z0.numel()).detach().numpy().astype(np.float64)
    if orthonormalize:
        basis, _ = np.linalg.qr(basis)
    return basis
