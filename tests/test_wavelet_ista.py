import numpy as np
import pytest
import torch

from mrsim.artifacts import decompose_error
from mrsim.data import random_ellipse_phantom
from mrsim.masks import variable_density_mask
from mrsim.recon import simulate_measurements, wavelet_ista

SIZE = 64


def _setup() -> tuple[torch.Tensor, np.ndarray, torch.Tensor]:
    truth = torch.from_numpy(random_ellipse_phantom(SIZE, np.random.default_rng(0)))
    mask = variable_density_mask(
        (SIZE, SIZE), (SIZE * SIZE) // 4, np.random.default_rng(1), n_center=20
    )
    y = simulate_measurements(truth[None], mask, noise_std=0.0)[0]
    return truth, mask, y


def test_imputes_nullspace_content():
    truth, mask, y = _setup()
    x = wavelet_ista(y, mask, threshold=0.02, n_iters=30)
    dec = decompose_error(x, truth, mask)
    assert dec.recon_nullspace_norm > 1e-3 * dec.truth_nullspace_norm
    assert dec.no_nullspace_content is False


def test_objective_history_nonincreasing():
    _, mask, y = _setup()
    result = wavelet_ista(y, mask, threshold=0.02, n_iters=30, return_history=True)
    history = result.objective_history
    assert len(history) == 30
    for previous, current in zip(history, history[1:]):
        assert current <= previous * (1.0 + 1e-6) + 1e-8


def test_plain_ista_history_is_also_nonincreasing():
    _, mask, y = _setup()
    result = wavelet_ista(
        y, mask, threshold=0.02, n_iters=20, momentum=False, return_history=True
    )
    history = result.objective_history
    assert len(history) == 20
    for previous, current in zip(history, history[1:]):
        assert current <= previous * (1.0 + 1e-6) + 1e-8


def test_momentum_reaches_a_lower_objective_than_plain_ista():
    """The point of the accelerated solver: mask comparison must not be
    confounded by how far from the minimizer the solver stopped."""
    _, mask, y = _setup()
    kwargs = dict(threshold=0.02, final_dc=False, return_history=True)
    plain = wavelet_ista(y, mask, n_iters=30, momentum=False, **kwargs)
    fast = wavelet_ista(y, mask, n_iters=30, momentum=True, **kwargs)
    assert fast.objective_history[-1] < plain.objective_history[-1]


def test_tolerance_stops_early_without_changing_the_contract():
    _, mask, y = _setup()
    result = wavelet_ista(
        y, mask, threshold=0.02, n_iters=500, tol=1e-4, return_history=True
    )
    assert len(result.objective_history) < 500


def test_negative_tolerance_is_rejected():
    _, mask, y = _setup()
    with pytest.raises(ValueError):
        wavelet_ista(y, mask, threshold=0.02, n_iters=2, tol=-1.0)


def test_final_dc_zeroes_consistency_error():
    truth, mask, y = _setup()
    x = wavelet_ista(y, mask, threshold=0.02, n_iters=10, final_dc=True)
    dec = decompose_error(x, truth, mask)
    truth_norm = torch.linalg.vector_norm(truth.to(torch.complex64)).item()
    assert dec.consistency_norm <= 1e-3 * truth_norm
