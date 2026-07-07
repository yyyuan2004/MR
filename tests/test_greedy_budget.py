import numpy as np
import pytest

from mrsim.greedy import greedy_a_optimal, greedy_artifact_aware, greedy_data_driven

SHAPE = (16, 16)


def _spectrum() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.random(SHAPE) + 0.1


def _check(mask: np.ndarray, n_samples: int) -> None:
    assert mask.shape == SHAPE
    assert set(np.unique(mask)).issubset({0.0, 1.0})
    assert int(mask.sum()) == n_samples


@pytest.mark.parametrize("n_samples", [1, 20, 64])
def test_a_optimal_budget(n_samples):
    _check(greedy_a_optimal(_spectrum(), n_samples), n_samples)


@pytest.mark.parametrize("n_samples", [4, 20, 40])
def test_artifact_aware_budget(n_samples):
    mask = greedy_artifact_aware(_spectrum(), n_samples, n_candidates=8)
    _check(mask, n_samples)


@pytest.mark.parametrize("n_samples", [1, 20, 64])
def test_data_driven_budget(n_samples):
    images = np.random.default_rng(1).random((5, *SHAPE))
    _check(greedy_data_driven(images, n_samples), n_samples)


def test_center_preselection_counts_toward_budget():
    mask = greedy_a_optimal(_spectrum(), 20, n_center=6)
    _check(mask, 20)
    assert mask[SHAPE[0] // 2, SHAPE[1] // 2] == 1.0


def test_invalid_budget_raises():
    with pytest.raises(ValueError):
        greedy_a_optimal(_spectrum(), 0)
    with pytest.raises(ValueError):
        greedy_data_driven(np.random.default_rng(0).random((3, *SHAPE)), SHAPE[0] * SHAPE[1] + 1)
