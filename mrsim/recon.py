"""Measurement simulation and simple reconstruction methods."""

from __future__ import annotations

import math

import numpy as np
import torch

from .fft_ops import as_mask_tensor, fft2c, ifft2c


def simulate_measurements(
    images: torch.Tensor,
    mask: np.ndarray | torch.Tensor,
    noise_std: float = 0.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Masked k-space measurements y = M (F x + n), with iid complex noise n."""
    k = fft2c(images.to(torch.complex64))
    if noise_std > 0.0:
        real = torch.randn(k.shape, generator=generator)
        imag = torch.randn(k.shape, generator=generator)
        # Total complex variance noise_std^2, split evenly between components.
        k = k + (noise_std / math.sqrt(2.0)) * (real + 1j * imag)
    return as_mask_tensor(mask) * k


def zero_filled(y: torch.Tensor) -> torch.Tensor:
    """Zero-filled reconstruction F^H y of already-masked k-space data."""
    return ifft2c(y)


def ridge(
    y: torch.Tensor,
    mask: np.ndarray | torch.Tensor,
    lam: float,
    spectrum: np.ndarray | torch.Tensor | None = None,
) -> torch.Tensor:
    """Ridge / Wiener reconstruction, solved per k-space coefficient.

    argmin_x ||M F x - y||^2 + lam ||x||^2 decouples in k-space because F is
    unitary and M is diagonal: sampled coefficients shrink by 1/(1 + lam) and
    unsampled coefficients are zero. With a diagonal prior power spectrum s_k,
    the shrinkage becomes the Wiener weight s_k / (s_k + lam).
    """
    if lam < 0.0:
        raise ValueError("lam must be non-negative")
    mask_t = as_mask_tensor(mask)
    if spectrum is None:
        weight: torch.Tensor | float = 1.0 / (1.0 + lam)
    else:
        s = torch.as_tensor(np.asarray(spectrum), dtype=torch.float32) if isinstance(
            spectrum, np.ndarray
        ) else spectrum.to(torch.float32)
        weight = s / (s + lam)
    return ifft2c(mask_t * weight * y)
