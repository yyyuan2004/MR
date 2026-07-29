import numpy as np
import pytest

from mrsim.masks import (
    budget_from_fraction,
    equispaced_lines_mask,
    fill_full_lines,
    line_count_from_sample_budget,
    uniform_random_mask,
    variable_density_mask,
)

SHAPE = (16, 16)
BUDGETS = [1, 16, 37, 100, 128, 256]
LINE_BUDGETS = [16, 32, 96, 128, 256]


def _check(mask: np.ndarray, n_samples: int) -> None:
    assert mask.shape == SHAPE
    assert set(np.unique(mask)).issubset({0.0, 1.0})
    assert int(mask.sum()) == n_samples


@pytest.mark.parametrize("n_samples", BUDGETS)
def test_uniform_random_budget(n_samples):
    mask = uniform_random_mask(SHAPE, n_samples, np.random.default_rng(0))
    _check(mask, n_samples)


@pytest.mark.parametrize("n_samples", BUDGETS)
def test_variable_density_budget(n_samples):
    mask = variable_density_mask(SHAPE, n_samples, np.random.default_rng(0))
    _check(mask, n_samples)


@pytest.mark.parametrize("n_samples", LINE_BUDGETS)
def test_equispaced_lines_budget(n_samples):
    mask = equispaced_lines_mask(SHAPE, n_samples)
    _check(mask, n_samples)
    assert np.all(mask.sum(axis=0) % SHAPE[0] == 0)


def test_nondivisible_line_budget_leaves_remainder_unspent():
    mask = equispaced_lines_mask(SHAPE, 100)
    assert line_count_from_sample_budget(SHAPE, 100) == 6
    _check(mask, 96)
    assert np.all(np.isin(mask.sum(axis=0), [0, SHAPE[0]]))


def test_explicit_full_line_api():
    priority = np.array([8, 7, 9, 6])
    mask = fill_full_lines(SHAPE, priority, n_lines=3)
    _check(mask, 3 * SHAPE[0])
    assert np.array_equal(np.flatnonzero(mask.sum(axis=0)), np.sort(priority[:3]))


def test_center_is_forced():
    mask = uniform_random_mask(SHAPE, 32, np.random.default_rng(0), n_center=4)
    assert mask[SHAPE[0] // 2, SHAPE[1] // 2] == 1.0
    _check(mask, 32)


def test_budget_from_fraction():
    assert budget_from_fraction(SHAPE, 0.25) == 64
    assert budget_from_fraction(SHAPE, 0.0) == 1  # clamped to at least one sample
    assert budget_from_fraction(SHAPE, 1.0) == 256


def test_invalid_budget_raises():
    with pytest.raises(ValueError):
        uniform_random_mask(SHAPE, 0, np.random.default_rng(0))
    with pytest.raises(ValueError):
        equispaced_lines_mask(SHAPE, 257)
    with pytest.raises(ValueError, match="complete Cartesian column"):
        equispaced_lines_mask(SHAPE, SHAPE[0] - 1)
