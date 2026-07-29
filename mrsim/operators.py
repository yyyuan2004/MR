"""Pluggable measurement operators.

The null-space decomposition only needs the forward operator A, its adjoint,
and an explicitly supplied orthogonal projector onto the row/observable
subspace, P_row = A† A. For a general operator this projector is not
A^H A. The equality P_row = A^H A holds for the partial-isometry operators
shipped here, whose nonzero singular values are one. Anything
satisfying that contract works verbatim — the decomposition, the artifact
field, and the leakage metrics are not tied to the Fourier case.

Two operators ship here:

- FourierOperator: A = M F, the frequency-domain sampling used everywhere
  else in this codebase (the default; identical to fft_ops.forward_op).
- InpaintingOperator: A = M, a pixel-domain 0/1 selection. P is then a
  diagonal projector in the signal domain itself, so the null space is
  literally the unobserved pixels.

Both expose orthogonal row-space projections (P^H = P, P^2 = P), which is the
property the decomposition relies on.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import torch

from .fft_ops import as_mask_tensor, fft2c, ifft2c


@runtime_checkable
class MeasurementOperator(Protocol):
    """Forward/adjoint pair plus the row-space projector P = A† A.

    ``projector`` is explicit rather than inferred as ``adjoint(forward(x))``:
    those expressions coincide only for a partial isometry.
    """

    name: str

    def forward(self, x: torch.Tensor, mask: np.ndarray | torch.Tensor) -> torch.Tensor: ...

    def adjoint(self, y: torch.Tensor, mask: np.ndarray | torch.Tensor) -> torch.Tensor: ...

    def projector(self, x: torch.Tensor, mask: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Apply the orthogonal projector onto the observable row space."""
        ...


class FourierOperator:
    """A = M F with binary M; here A^H A = A† A = F^H M F."""

    name = "fourier"
    #: Domain the mask indexes: measurements live in the frequency domain.
    measurement_domain = "frequency"

    def forward(self, x: torch.Tensor, mask: np.ndarray | torch.Tensor) -> torch.Tensor:
        k = fft2c(x)
        return as_mask_tensor(mask, like=k) * k

    def adjoint(self, y: torch.Tensor, mask: np.ndarray | torch.Tensor) -> torch.Tensor:
        return ifft2c(as_mask_tensor(mask, like=y) * y)

    def projector(self, x: torch.Tensor, mask: np.ndarray | torch.Tensor) -> torch.Tensor:
        k = fft2c(x)
        return ifft2c(as_mask_tensor(mask, like=k) * k)


class InpaintingOperator:
    """A = M, a pixel-domain selection. Measures signal samples directly.

    F never appears and binary M is its own pseudoinverse, so the row-space
    projector P = A† A = M is diagonal in the signal domain: the
    observed subspace is the measured pixels and the null space is the
    unmeasured ones. Useful for showing that null-space structure — not the
    Fourier transform — governs when a prior helps.
    """

    name = "inpainting"
    measurement_domain = "signal"

    def forward(self, x: torch.Tensor, mask: np.ndarray | torch.Tensor) -> torch.Tensor:
        return as_mask_tensor(mask, like=x) * x

    def adjoint(self, y: torch.Tensor, mask: np.ndarray | torch.Tensor) -> torch.Tensor:
        return as_mask_tensor(mask, like=y) * y

    def projector(self, x: torch.Tensor, mask: np.ndarray | torch.Tensor) -> torch.Tensor:
        return as_mask_tensor(mask, like=x) * x


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
