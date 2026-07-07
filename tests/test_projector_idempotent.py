import numpy as np
import torch

from mrsim.fft_ops import projector
from mrsim.masks import uniform_random_mask


def _random_complex(shape: tuple[int, ...], seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.complex(torch.randn(shape, generator=g), torch.randn(shape, generator=g))


def _mask() -> np.ndarray:
    return uniform_random_mask((16, 16), 64, np.random.default_rng(0))


def test_projector_idempotent():
    x = _random_complex((16, 16), seed=0)
    mask = _mask()
    px = projector(x, mask)
    ppx = projector(px, mask)
    assert torch.allclose(ppx, px, atol=1e-5)


def test_projector_hermitian():
    # <P x, y> == <x, P y> for the orthogonal projection P = F^H M F.
    mask = _mask()
    x = _random_complex((16, 16), seed=1)
    y = _random_complex((16, 16), seed=2)
    lhs = torch.sum(projector(x, mask) * y.conj())
    rhs = torch.sum(x * projector(y, mask).conj())
    assert torch.allclose(lhs, rhs, atol=1e-4)
