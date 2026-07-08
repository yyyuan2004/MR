"""Subspace and manifold priors for sampling design and reconstruction.

The prior is abstracted as a pluggable N x d basis matrix B whose columns
span the model of plausible signals. A linear subspace uses the SVD basis
(B = U); a generator manifold uses the Jacobian of a fixed pre-trained
generator at a reference latent point (B = J_G(z0)). Both follow the same
selection and reconstruction code path. NumPy throughout; PyTorch is used
only at the generator boundary.
"""

from __future__ import annotations

import numpy as np
import torch

from .fft_ops import fft2c


def fit_subspace(
    train_images: np.ndarray | torch.Tensor, d: int
) -> tuple[np.ndarray, float]:
    """Fit a d-dimensional linear subspace to vectorized training images.

    Returns the image-domain orthonormal basis U (N x d, columns are the top
    right singular vectors of the data matrix) and the fraction of training
    energy the subspace captures.
    """
    if isinstance(train_images, torch.Tensor):
        train_images = train_images.numpy()
    images = np.asarray(train_images, dtype=np.float64)
    if images.ndim != 3:
        raise ValueError("train_images must have shape (n, H, W)")
    n = images.shape[0]
    data = images.reshape(n, -1)
    if not (1 <= d <= min(n, data.shape[1])):
        raise ValueError(f"d={d} must be in [1, {min(n, data.shape[1])}]")

    _, singular_values, vt = np.linalg.svd(data, full_matrices=False)
    basis = vt[:d].conj().T
    energy = singular_values**2
    energy_ratio = float(energy[:d].sum() / max(energy.sum(), 1e-30))
    return basis, energy_ratio


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
