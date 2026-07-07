import torch

from mrsim.fft_ops import fft2c, ifft2c


def _random_complex(shape: tuple[int, ...], seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.complex(torch.randn(shape, generator=g), torch.randn(shape, generator=g))


def test_fft_roundtrip_returns_input():
    x = _random_complex((3, 16, 16), seed=0)
    assert torch.allclose(ifft2c(fft2c(x)), x, atol=1e-5)


def test_fft_roundtrip_real_input():
    g = torch.Generator().manual_seed(1)
    x = torch.randn(2, 16, 16, generator=g)
    out = ifft2c(fft2c(x))
    assert torch.allclose(out.real, x, atol=1e-5)
    assert torch.allclose(out.imag, torch.zeros_like(x), atol=1e-5)


def test_ifft_is_adjoint_of_fft():
    # <F x, y> == <x, F^H y> for the orthonormal centered FFT.
    x = _random_complex((16, 16), seed=2)
    y = _random_complex((16, 16), seed=3)
    lhs = torch.sum(fft2c(x) * y.conj())
    rhs = torch.sum(x * ifft2c(y).conj())
    assert torch.allclose(lhs, rhs, atol=1e-4)
