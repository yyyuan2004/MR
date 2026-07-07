import numpy as np
import torch

from mrsim.artifacts import artifact_map, decompose_error
from mrsim.fft_ops import projector
from mrsim.masks import uniform_random_mask

SHAPE = (16, 16)


def _mask() -> np.ndarray:
    return uniform_random_mask(SHAPE, 64, np.random.default_rng(0))


def _random_complex(seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.complex(torch.randn(SHAPE, generator=g), torch.randn(SHAPE, generator=g))


def _random_real(seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(SHAPE, generator=g)


def test_exactness_identities():
    recon = _random_complex(0)
    truth = _random_real(1)
    dec = decompose_error(recon, truth, _mask())

    err = recon.to(torch.complex64) - truth.to(torch.complex64)
    assert torch.allclose(dec.consistency_error + dec.artifact_field, err, atol=1e-5, rtol=1e-4)
    # (I-P)recon - (I-P)(recon - truth) = (I-P)truth: algebraic identity.
    assert torch.allclose(
        dec.recon_nullspace - dec.artifact_field, dec.truth_nullspace, atol=1e-5, rtol=1e-4
    )


def test_observed_subspace_confined_case():
    truth = _random_real(2)
    mask = _mask()
    recon = projector(truth.to(torch.complex64), mask)
    dec = decompose_error(recon, truth, mask)

    zero = torch.zeros_like(dec.consistency_error)
    assert torch.allclose(dec.consistency_error, zero, atol=1e-5)
    assert torch.allclose(dec.recon_nullspace, zero, atol=1e-5)
    assert torch.allclose(dec.artifact_field, -dec.truth_nullspace, atol=1e-5, rtol=1e-4)
    assert dec.no_nullspace_content is True


def test_nullspace_filling_case():
    alpha = 0.5
    truth = _random_real(3)
    mask = _mask()
    truth_c = truth.to(torch.complex64)
    kept = projector(truth_c, mask)
    recon = kept + alpha * (truth_c - kept)
    dec = decompose_error(recon, truth, mask)

    assert torch.allclose(dec.consistency_error, torch.zeros_like(kept), atol=1e-5)
    assert torch.allclose(dec.recon_nullspace, alpha * dec.truth_nullspace, atol=1e-5, rtol=1e-4)
    assert torch.allclose(
        dec.artifact_field, (alpha - 1.0) * dec.truth_nullspace, atol=1e-5, rtol=1e-4
    )
    assert dec.no_nullspace_content is False


def test_artifact_map_backward_compatible():
    recon = _random_complex(4)
    truth = _random_real(5)
    expected = (recon.to(torch.complex64) - truth.to(torch.complex64)).abs()
    assert torch.allclose(artifact_map(recon, truth), expected)
    dec = decompose_error(recon, truth, _mask())
    assert torch.allclose(dec.total_error, expected)
