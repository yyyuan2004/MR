import numpy as np
import pytest

from mrsim.masks import (
    budget_from_fraction,
    equispaced_lines_mask,
    uniform_random_mask,
    variable_density_mask,
)

SHAPE = (16, 16)
# Budgets chosen to exercise remainder handling (100 = 6 full lines + 4 extra).
BUDGETS = [1, 16, 37, 100, 128, 256]


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


@pytest.mark.parametrize("n_samples", BUDGETS)
def test_equispaced_lines_budget(n_samples):
    mask = equispaced_lines_mask(SHAPE, n_samples)
    _check(mask, n_samples)


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
