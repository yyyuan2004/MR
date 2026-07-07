"""Measurement simulation and simple reconstruction methods."""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
import pywt
import torch

from .fft_ops import as_mask_tensor, fft2c, ifft2c, projector


def simulate_measurements(
    images: torch.Tensor,
    mask: np.ndarray | torch.Tensor,
    noise_std: float = 0.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Masked frequency-domain measurements y = M (F x + n), with iid complex noise n."""
    k = fft2c(images.to(torch.complex64))
    if noise_std > 0.0:
        real = torch.randn(k.shape, generator=generator)
        imag = torch.randn(k.shape, generator=generator)
        # Total complex variance noise_std^2, split evenly between components.
        k = k + (noise_std / math.sqrt(2.0)) * (real + 1j * imag)
    return as_mask_tensor(mask) * k


def zero_filled(y: torch.Tensor) -> torch.Tensor:
    """Zero-filled reconstruction F^H y of already-masked frequency-domain data."""
    return ifft2c(y)


def ridge(
    y: torch.Tensor,
    mask: np.ndarray | torch.Tensor,
    lam: float,
    spectrum: np.ndarray | torch.Tensor | None = None,
) -> torch.Tensor:
    """Ridge / Wiener reconstruction, solved per frequency-domain coefficient.

    argmin_x ||M F x - y||^2 + lam ||x||^2 decouples in the frequency domain
    because F is unitary and M is diagonal: measured coefficients shrink by
    1/(1 + lam) and unmeasured coefficients are zero. With a diagonal prior
    power spectrum s_k, the shrinkage becomes the Wiener weight
    s_k / (s_k + lam). Either way the estimator's unmeasured coefficients stay
    at the prior mean (zero), so the reconstruction remains confined to the
    observed subspace and imputes no null-space content.
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


class IstaResult(NamedTuple):
    """Wavelet-ISTA output with the per-iteration objective values."""

    image: torch.Tensor
    objective_history: list[float]


def _wavelet_decompose(x: np.ndarray, wavelet: str, levels: int) -> list:
    return pywt.wavedec2(x, wavelet=wavelet, mode="periodization", level=levels, axes=(-2, -1))


def _wavelet_soft_threshold(
    x: torch.Tensor, threshold: float, wavelet: str, levels: int
) -> torch.Tensor:
    """Complex-magnitude soft-thresholding in the wavelet domain.

    Shrinks each complex coefficient by max(1 - t/|c|, 0), which preserves the
    phase; real and imaginary parts share one decomposition (PyWavelets
    transforms complex input componentwise).
    """
    arr = x.numpy().astype(np.complex128)
    coeffs = _wavelet_decompose(arr, wavelet, levels)

    def shrink(c: np.ndarray) -> np.ndarray:
        mag = np.abs(c)
        return c * np.maximum(1.0 - threshold / np.maximum(mag, 1e-12), 0.0)

    shrunk = [shrink(coeffs[0])] + [tuple(shrink(d) for d in band) for band in coeffs[1:]]
    out = pywt.waverec2(shrunk, wavelet=wavelet, mode="periodization", axes=(-2, -1))
    return torch.from_numpy(out.astype(np.complex64))


def _wavelet_l1(x: torch.Tensor, wavelet: str, levels: int) -> float:
    coeffs = _wavelet_decompose(x.numpy().astype(np.complex128), wavelet, levels)
    total = float(np.abs(coeffs[0]).sum())
    for band in coeffs[1:]:
        total += float(sum(np.abs(d).sum() for d in band))
    return total


def wavelet_ista(
    y: torch.Tensor,
    mask: np.ndarray | torch.Tensor,
    *,
    threshold: float,
    n_iters: int = 50,
    wavelet: str = "db4",
    levels: int = 3,
    final_dc: bool = True,
    return_history: bool = False,
) -> torch.Tensor | IstaResult:
    """Solve min_x 0.5 * ||M F x - y||^2 + threshold * ||W x||_1 by ISTA.

    Step size 1.0 is valid because the forward operator satisfies A^H A = P
    (an orthogonal projector), so the data-fidelity gradient is 1-Lipschitz.
    Thresholding in the wavelet basis couples frequency coefficients, so
    iterates leave the observed subspace and impute null-space content —
    unlike linear diagonal reconstructions, which cannot.

    If final_dc, one data-consistency step replaces measured frequency
    coefficients with the measurements. Deterministic: no randomness anywhere.
    With return_history, returns IstaResult(image, objective_history); the
    history holds the objective after each proximal step and is non-increasing
    up to floating-point round-off.
    """
    if threshold < 0.0:
        raise ValueError("threshold must be non-negative")
    mask_t = as_mask_tensor(mask)
    y = y.to(torch.complex64)
    adjoint = ifft2c(y)  # A^H y; y is already masked
    x = adjoint.clone()
    history: list[float] = []
    for _ in range(n_iters):
        # Gradient of the data term is A^H (A x - y) = P x - A^H y.
        x_grad = x - (projector(x, mask_t) - adjoint)
        x = _wavelet_soft_threshold(x_grad, threshold, wavelet, levels)
        if return_history:
            residual = mask_t * fft2c(x) - y
            objective = 0.5 * float((residual.abs() ** 2).sum()) + threshold * _wavelet_l1(
                x, wavelet, levels
            )
            history.append(objective)
    if final_dc:
        x = x + ifft2c(y - mask_t * fft2c(x))
    if return_history:
        return IstaResult(image=x, objective_history=history)
    return x
