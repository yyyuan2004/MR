import numpy as np
import pytest
import torch

from mrsim.greedy import (
    _aopt_gain,
    _sidelobe_reduction_candidates,
    greedy_a_optimal,
)
from mrsim.operators import MeasurementOperator
from mrsim.subspace import SubspaceStatistics, fit_subspace


def _ifft2c_numpy(kspace: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(
        np.fft.ifft2(np.fft.ifftshift(kspace), norm="ortho")
    )


def test_pointwise_aopt_is_noise_invariant_top_k():
    spectrum = np.array(
        [
            [4.0, 4.0, 0.0, 1.0],
            [3.0, 2.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0, 0.0],
        ]
    )
    masks = [
        greedy_a_optimal(spectrum, 6, noise_var=noise_var)
        for noise_var in (0.0, 1e-6, 1.0, 1e12)
    ]
    assert all(np.array_equal(mask, masks[0]) for mask in masks[1:])

    expected = np.argsort(-spectrum.ravel(), kind="stable")[:6]
    assert np.array_equal(np.flatnonzero(masks[0]), np.sort(expected))


def test_pointwise_aopt_zero_spectrum_and_sigma_zero_are_stable():
    spectrum = np.zeros((4, 4))
    gain = _aopt_gain(spectrum, noise_var=0.0)
    assert np.array_equal(gain, np.zeros(spectrum.size))
    assert np.all(np.isfinite(gain))

    mask = greedy_a_optimal(spectrum, 5, noise_var=0.0)
    assert np.array_equal(np.flatnonzero(mask), np.arange(5))


@pytest.mark.parametrize("noise_var", [-1.0, np.inf, np.nan])
def test_pointwise_aopt_rejects_invalid_noise(noise_var):
    with pytest.raises(ValueError, match="noise_var"):
        greedy_a_optimal(np.ones((4, 4)), 4, noise_var=noise_var)


@pytest.mark.parametrize("shape", [(8, 8), (9, 9)])
def test_sidelobe_candidates_match_exact_centered_ifft(shape):
    rng = np.random.default_rng(sum(shape))
    mask = np.zeros(shape, dtype=np.float64)
    chosen = rng.choice(mask.size, size=7, replace=False)
    mask.ravel()[chosen] = 1.0
    selected = mask.ravel().astype(bool)
    weight = rng.uniform(0.2, 1.7, size=shape)
    n_wanted = 5

    actual = _sidelobe_reduction_candidates(
        mask, selected, weight, n_wanted=n_wanted
    )

    psf = _ifft2c_numpy(mask * weight)
    magnitude = np.abs(psf)
    center = (shape[0] // 2, shape[1] // 2)
    magnitude[center] = 0.0
    target = np.unravel_index(int(np.argmax(magnitude)), shape)

    reduction = np.full(mask.size, -np.inf)
    for candidate in np.flatnonzero(~selected):
        delta = np.zeros(shape)
        delta.ravel()[candidate] = weight.ravel()[candidate]
        contribution = _ifft2c_numpy(delta)[target]
        reduction[candidate] = -np.real(np.conj(psf[target]) * contribution)
    expected = np.argpartition(reduction, -n_wanted)[-n_wanted:]

    assert set(actual.tolist()) == set(expected.tolist())


def test_centered_subspace_returns_mean_and_covariance_eigenvalues():
    mean = np.array([10.0, -3.0, 2.0, 0.5])
    centered = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, -2.0, 0.0, 0.0],
            [0.0, 0.0, 3.0, 0.0],
            [0.0, 0.0, -3.0, 0.0],
        ]
    )
    images = (mean + centered).reshape(-1, 2, 2)

    stats = fit_subspace(
        images, 2, center=True, return_statistics=True
    )
    assert isinstance(stats, SubspaceStatistics)
    assert stats.centered is True
    assert np.allclose(stats.mean, mean)
    assert np.allclose(stats.mean_image, mean.reshape(2, 2))
    assert np.allclose(stats.eigenvalues, [18.0 / 5.0, 8.0 / 5.0])
    assert np.isclose(stats.total_variance, 28.0 / 5.0)
    assert np.isclose(stats.energy_ratio, 26.0 / 28.0)
    assert np.allclose(stats.basis.conj().T @ stats.basis, np.eye(2))
    assert np.allclose(
        stats.covariance_factor @ stats.covariance_factor.conj().T,
        stats.basis @ np.diag(stats.eigenvalues) @ stats.basis.conj().T,
    )


def test_legacy_subspace_return_contract_is_preserved():
    images = np.arange(6 * 2 * 2, dtype=np.float64).reshape(6, 2, 2)
    basis, ratio = fit_subspace(images, 2)
    stats = fit_subspace(images, 2, return_statistics=True)

    assert isinstance(stats, SubspaceStatistics)
    assert stats.centered is False
    assert np.allclose(
        basis @ basis.conj().T,
        stats.basis @ stats.basis.conj().T,
    )
    assert ratio == stats.energy_ratio


class _ScaledCoordinateOperator:
    """A non-partial-isometry example with A^H A != A† A."""

    name = "scaled_coordinate"

    def forward(self, x, mask):
        return torch.stack((2.0 * x[0], torch.zeros_like(x[1])))

    def adjoint(self, y, mask):
        return torch.stack((2.0 * y[0], torch.zeros_like(y[1])))

    def projector(self, x, mask):
        return torch.stack((x[0], torch.zeros_like(x[1])))


def test_operator_contract_uses_explicit_row_space_projector():
    operator = _ScaledCoordinateOperator()
    assert isinstance(operator, MeasurementOperator)
    x = torch.tensor([3.0, 5.0])
    mask = np.ones(2)

    projected = operator.projector(x, mask)
    normal_action = operator.adjoint(operator.forward(x, mask), mask)
    assert torch.equal(projected, torch.tensor([3.0, 0.0]))
    assert torch.equal(normal_action, torch.tensor([12.0, 0.0]))
    assert not torch.equal(projected, normal_action)
