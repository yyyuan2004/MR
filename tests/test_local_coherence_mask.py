"""Local-coherence sampling density, and why global coherence is the wrong tool."""

from __future__ import annotations

import numpy as np
import pytest

from mrsim import artifacts, masks

SHAPE = (64, 64)


def test_global_coherence_bound_is_vacuous_here():
    """m >~ mu^2 k log N exceeds the ambient dimension, so it says nothing."""
    coherence_squared = artifacts.local_coherence_map(SHAPE, "db4", 3)
    mu = float(np.sqrt(coherence_squared.max()))
    n = SHAPE[0] * SHAPE[1]
    assert mu > 4.0, "Fourier and wavelets are far from incoherent"
    required = mu**2 * 256 * np.log(n)
    assert required > n, "if this bound were informative the theory would apply directly"


def test_coherence_varies_by_orders_across_frequency():
    """The variation is the usable signal; a single maximum is not."""
    coherence_squared = artifacts.local_coherence_map(SHAPE, "db4", 3)
    radius = masks.radius_map(SHAPE)
    centre = coherence_squared[radius < 6].mean()
    edge = coherence_squared[radius > 24].mean()
    assert centre > 4.0 * edge


def test_density_meets_the_budget_and_stays_binary():
    coherence = artifacts.local_coherence_map(SHAPE, "db4", 3)
    for budget in (256, 1024, 2048):
        mask = masks.coherence_weighted_mask(
            SHAPE, budget, np.random.default_rng(0), coherence, n_center=8
        )
        assert int(mask.sum()) == budget
        assert set(np.unique(mask)) <= {0.0, 1.0}


def test_density_concentrates_where_coherence_is_high():
    coherence = artifacts.local_coherence_map(SHAPE, "db4", 3)
    mask = masks.coherence_weighted_mask(
        SHAPE, 1024, np.random.default_rng(0), coherence
    )
    radius = masks.radius_map(SHAPE)
    uniform = masks.uniform_random_mask(SHAPE, 1024, np.random.default_rng(0))
    assert (mask * radius).sum() / mask.sum() < (uniform * radius).sum() / uniform.sum()


def test_budget_larger_than_the_support_still_fills_exactly():
    weights = np.zeros(SHAPE)
    weights[:4, :4] = 1.0
    mask = masks.coherence_weighted_mask(SHAPE, 100, np.random.default_rng(0), weights)
    assert int(mask.sum()) == 100


def test_rejects_bad_weights():
    with pytest.raises(ValueError):
        masks.coherence_weighted_mask(
            SHAPE, 64, np.random.default_rng(0), -np.ones(SHAPE)
        )
    with pytest.raises(ValueError):
        masks.coherence_weighted_mask(
            SHAPE, 64, np.random.default_rng(0), np.ones((8, 8))
        )
