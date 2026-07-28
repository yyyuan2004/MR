import numpy as np
import pytest
import torch

from mrsim.artifacts import decompose_error
from mrsim.fft_ops import projector as fourier_projector
from mrsim.masks import uniform_random_mask
from mrsim.operators import FOURIER, INPAINTING, get_operator

SHAPE = (16, 16)
OPERATORS = [FOURIER, INPAINTING]


def _mask() -> np.ndarray:
    return uniform_random_mask(SHAPE, 64, np.random.default_rng(0))


def _random_complex(seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.complex(torch.randn(SHAPE, generator=g), torch.randn(SHAPE, generator=g))


def test_registry_lookup():
    assert get_operator("fourier") is FOURIER
    assert get_operator("inpainting") is INPAINTING
    with pytest.raises(ValueError):
        get_operator("nope")


def test_fourier_operator_matches_existing_projector():
    # The default path must be bit-for-bit the pre-existing behavior.
    x = _random_complex(0)
    mask = _mask()
    assert torch.allclose(FOURIER.projector(x, mask), fourier_projector(x, mask))


@pytest.mark.parametrize("operator", OPERATORS, ids=lambda o: o.name)
def test_projector_idempotent_and_hermitian(operator):
    mask = _mask()
    x, y = _random_complex(1), _random_complex(2)
    px = operator.projector(x, mask)
    assert torch.allclose(operator.projector(px, mask), px, atol=1e-5)
    lhs = torch.sum(operator.projector(x, mask) * y.conj())
    rhs = torch.sum(x * operator.projector(y, mask).conj())
    assert torch.allclose(lhs, rhs, atol=1e-4)


@pytest.mark.parametrize("operator", OPERATORS, ids=lambda o: o.name)
def test_adjoint_identity(operator):
    # <A x, y> == <x, A^H y>
    mask = _mask()
    x, y = _random_complex(3), _random_complex(4)
    lhs = torch.sum(operator.forward(x, mask) * y.conj())
    rhs = torch.sum(x * operator.adjoint(y, mask).conj())
    assert torch.allclose(lhs, rhs, atol=1e-4)


@pytest.mark.parametrize("operator", OPERATORS, ids=lambda o: o.name)
def test_decomposition_identities_hold_for_any_operator(operator):
    mask = _mask()
    recon, truth = _random_complex(5), _random_complex(6)
    dec = decompose_error(recon, truth, mask, operator=operator)
    err = recon - truth
    assert torch.allclose(dec.consistency_error + dec.artifact_field, err, atol=1e-5, rtol=1e-4)
    assert torch.allclose(
        dec.recon_nullspace - dec.artifact_field, dec.truth_nullspace, atol=1e-5, rtol=1e-4
    )
    # Pythagorean split: ||e||^2 = ||Pe||^2 + ||(I-P)e||^2.
    total = dec.total_error_norm**2
    assert np.isclose(dec.consistency_norm**2 + dec.artifact_norm**2, total, rtol=1e-4)


def test_inpainting_nullspace_is_exactly_unmeasured_pixels():
    mask = _mask()
    x = _random_complex(7)
    dec = decompose_error(x, torch.zeros(SHAPE), mask, operator=INPAINTING)
    unmeasured = torch.from_numpy(1.0 - mask).to(torch.complex64)
    assert torch.allclose(dec.recon_nullspace, x * unmeasured, atol=1e-6)
    # Measured pixels carry no null-space content at all.
    assert torch.allclose(
        dec.recon_nullspace * torch.from_numpy(mask), torch.zeros(SHAPE, dtype=torch.complex64)
    )


def test_default_operator_is_fourier():
    mask = _mask()
    recon, truth = _random_complex(8), _random_complex(9)
    assert torch.allclose(
        decompose_error(recon, truth, mask).artifact_field,
        decompose_error(recon, truth, mask, operator=FOURIER).artifact_field,
    )
