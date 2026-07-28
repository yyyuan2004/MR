"""Pluggable measurement operators.

The null-space decomposition only needs the forward operator A, its adjoint,
and the orthogonal projector P = A^H A onto the observed subspace. Anything
satisfying that contract works verbatim — the decomposition, the artifact
field, and the leakage metrics are not tied to the Fourier case.

Two operators ship here:

- FourierOperator: A = M F, the frequency-domain sampling used everywhere
  else in this codebase (the default; identical to fft_ops.forward_op).
- InpaintingOperator: A = M, a pixel-domain 0/1 selection. P is then a
  diagonal projector in the signal domain itself, so the null space is
  literally the unobserved pixels.

Both are orthogonal projections (P^H = P, P^2 = P), which is the only
property the decomposition relies on.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import torch

from .fft_ops import as_mask_tensor, fft2c, ifft2c


@runtime_checkable
class MeasurementOperator(Protocol):
    """Forward operator A, its adjoint A^H, and the projector P = A^H A."""

    name: str

    def forward(self, x: torch.Tensor, mask: np.ndarray | torch.Tensor) -> torch.Tensor: ...

    def adjoint(self, y: torch.Tensor, mask: np.ndarray | torch.Tensor) -> torch.Tensor: ...

    def projector(self, x: torch.Tensor, mask: np.ndarray | torch.Tensor) -> torch.Tensor: ...


class FourierOperator:
    """A = M F with the centered orthonormal transform. Measures frequencies."""

    name = "fourier"
    #: Domain the mask indexes: measurements live in the frequency domain.
    measurement_domain = "frequency"

    def forward(self, x: torch.Tensor, mask: np.ndarray | torch.Tensor) -> torch.Tensor:
        return as_mask_tensor(mask) * fft2c(x)

    def adjoint(self, y: torch.Tensor, mask: np.ndarray | torch.Tensor) -> torch.Tensor:
        return ifft2c(as_mask_tensor(mask) * y)

    def projector(self, x: torch.Tensor, mask: np.ndarray | torch.Tensor) -> torch.Tensor:
        return ifft2c(as_mask_tensor(mask) * fft2c(x))


class InpaintingOperator:
    """A = M, a pixel-domain selection. Measures signal samples directly.

    F never appears, so P = M is already diagonal in the signal domain: the
    observed subspace is the measured pixels and the null space is the
    unmeasured ones. Useful for showing that null-space structure — not the
    Fourier transform — governs when a prior helps.
    """

    name = "inpainting"
    measurement_domain = "signal"

    def forward(self, x: torch.Tensor, mask: np.ndarray | torch.Tensor) -> torch.Tensor:
        return as_mask_tensor(mask) * x

    def adjoint(self, y: torch.Tensor, mask: np.ndarray | torch.Tensor) -> torch.Tensor:
        return as_mask_tensor(mask) * y

    def projector(self, x: torch.Tensor, mask: np.ndarray | torch.Tensor) -> torch.Tensor:
        return as_mask_tensor(mask) * x


FOURIER = FourierOperator()
INPAINTING = InpaintingOperator()

_REGISTRY: dict[str, MeasurementOperator] = {
    FOURIER.name: FOURIER,
    INPAINTING.name: INPAINTING,
}


def get_operator(name: str) -> MeasurementOperator:
    """Look up a measurement operator by name."""
    if name not in _REGISTRY:
        raise ValueError(f"unknown operator {name!r}; known: {sorted(_REGISTRY)}")
    return _REGISTRY[name]
