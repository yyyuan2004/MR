"""Centered orthonormal 2D FFT operators and the sampling projector."""

from __future__ import annotations

import numpy as np
import torch


def fft2c(x: torch.Tensor) -> torch.Tensor:
    """Centered orthonormal 2D FFT over the last two dimensions."""
    x = torch.fft.ifftshift(x, dim=(-2, -1))
    k = torch.fft.fft2(x, dim=(-2, -1), norm="ortho")
    return torch.fft.fftshift(k, dim=(-2, -1))


def ifft2c(k: torch.Tensor) -> torch.Tensor:
    """Centered orthonormal 2D inverse FFT over the last two dimensions."""
    k = torch.fft.ifftshift(k, dim=(-2, -1))
    x = torch.fft.ifft2(k, dim=(-2, -1), norm="ortho")
    return torch.fft.fftshift(x, dim=(-2, -1))


def as_mask_tensor(mask: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Convert a mask to a float32 tensor suitable for broadcasting."""
    if isinstance(mask, np.ndarray):
        mask = torch.from_numpy(np.ascontiguousarray(mask))
    return mask.to(torch.float32)


def forward_op(x: torch.Tensor, mask: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Forward operator A = M F: masked k-space of an image."""
    return as_mask_tensor(mask) * fft2c(x)


def adjoint_op(y: torch.Tensor, mask: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Adjoint operator A^H = F^H M applied to k-space data."""
    return ifft2c(as_mask_tensor(mask) * y)


def projector(x: torch.Tensor, mask: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Apply P = F^H M F.

    F is unitary and M is a diagonal 0/1 selection, so P is an orthogonal
    projection: Hermitian (P^H = P) and idempotent (P^2 = P).
    """
    return ifft2c(as_mask_tensor(mask) * fft2c(x))
