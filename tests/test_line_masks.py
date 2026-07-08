import numpy as np
import pytest

from mrsim.greedy import (
    greedy_line_a_optimal,
    greedy_lines_recon_in_loop,
    greedy_lines_spectrum_energy,
    greedy_lines_subspace_leakage,
    greedy_psf_penalized_aopt,
)
from mrsim.masks import jaccard, multilevel_random_mask, variable_density_lines_mask

SHAPE = (16, 16)
BUDGETS = [1, 16, 37, 100, 128, 256]


def _check(mask: np.ndarray, n_samples: int) -> None:
    assert mask.shape == SHAPE
    assert set(np.unique(mask)).issubset({0.0, 1.0})
    assert int(mask.sum()) == n_samples


def _images(n: int = 6) -> np.ndarray:
    return np.random.default_rng(0).random((n, *SHAPE))


@pytest.mark.parametrize("n_samples", BUDGETS)
def test_variable_density_lines_budget(n_samples):
    mask = variable_density_lines_mask(SHAPE, n_samples, np.random.default_rng(0))
    _check(mask, n_samples)


@pytest.mark.parametrize("n_samples", BUDGETS)
def test_multilevel_random_budget(n_samples):
    mask = multilevel_random_mask(SHAPE, n_samples, np.random.default_rng(0))
    _check(mask, n_samples)


@pytest.mark.parametrize("n_samples", [16, 37, 100])
def test_line_aopt_budget(n_samples):
    spectrum = np.random.default_rng(1).random(SHAPE) + 0.1
    _check(greedy_line_a_optimal(spectrum, n_samples), n_samples)


@pytest.mark.parametrize("n_samples", [16, 37, 100])
def test_spectrum_energy_lines_budget(n_samples):
    _check(greedy_lines_spectrum_energy(_images(), n_samples), n_samples)


@pytest.mark.parametrize("n_samples", [16, 37, 100])
def test_subspace_leakage_lines_budget(n_samples):
    mask = greedy_lines_subspace_leakage(_images(), n_samples, wavelet="db2", levels=2)
    _check(mask, n_samples)


def test_recon_in_loop_budget():
    mask = greedy_lines_recon_in_loop(
        _images(4),
        64,
        n_candidate_lines=4,
        batch_size=2,
        ista_iters=2,
        wavelet="db2",
        levels=2,
        rng=np.random.default_rng(0),
    )
    _check(mask, 64)


def test_center_lines_are_included():
    mask = variable_density_lines_mask(SHAPE, 64, np.random.default_rng(0), n_center_lines=2)
    assert mask[:, SHAPE[1] // 2].sum() == SHAPE[0]


def test_psf_penalty_changes_selection():
    # With a hybrid candidate pool and a nonzero penalty, the selection must
    # differ from plain top-gain behavior (guards against the degenerate
    # near-1.0 Jaccard overlap between the penalized and plain variants).
    rng = np.random.default_rng(2)
    r = np.hypot(*np.meshgrid(np.arange(16) - 8, np.arange(16) - 8, indexing="ij"))
    spectrum = (1.0 + r) ** (-3.0)
    unpenalized = greedy_psf_penalized_aopt(
        spectrum, 64, beta=0.0, n_candidates=16, rng=np.random.default_rng(3)
    )
    penalized = greedy_psf_penalized_aopt(
        spectrum, 64, beta=4.0, n_candidates=16, rng=np.random.default_rng(3)
    )
    assert int(unpenalized.sum()) == 64
    assert int(penalized.sum()) == 64
    assert jaccard(unpenalized, penalized) < 0.999
