"""Exact Gaussian design quantities, checked against brute force.

These tests pin two facts that the rest of the package depends on:

- under the independent-coefficient model a top-k ranking of the prior
  spectrum is exactly the optimal design, so there is no design problem to
  solve in that regime; and
- once the signal is modelled as real, that is no longer true, because a
  conjugate pair is one unknown rather than two.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from mrsim import design
from mrsim.artifacts import conjugate_reflect
from mrsim.greedy import greedy_a_optimal, greedy_a_optimal_hermitian

NOISE_VAR = 0.05


def _symmetric_spectrum(shape, seed=3):
    """A real signal's power spectrum is conjugate symmetric."""
    rng = np.random.default_rng(seed)
    power = rng.random(shape) + 0.1
    return power + conjugate_reflect(power)


def _exhaustive_optimum(power, budget, noise_var, hermitian):
    n = power.size
    best = None
    for indices in itertools.combinations(range(n), budget):
        flat = np.zeros(n)
        flat[list(indices)] = 1.0
        mask = flat.reshape(power.shape)
        value = design.diagonal_bayes(
            mask, power, noise_var, hermitian=hermitian
        )["bayes_mmse_total"]
        if best is None or value < best[0]:
            best = (value, mask)
    return best


def test_independent_model_reduces_to_top_k():
    """The documented degeneracy: with independent coefficients, design is top-k."""
    power = _symmetric_spectrum((4, 4))
    budget = 6
    optimum, _ = _exhaustive_optimum(power, budget, NOISE_VAR, hermitian=False)
    top_k = design.diagonal_bayes(
        greedy_a_optimal(power, budget), power, NOISE_VAR, hermitian=False
    )["bayes_mmse_total"]
    assert top_k == pytest.approx(optimum, rel=1e-12)


def test_top_k_is_badly_suboptimal_for_a_real_signal():
    """Top-k buys both members of each conjugate pair and wastes the budget."""
    power = _symmetric_spectrum((4, 4))
    budget = 6
    optimum, _ = _exhaustive_optimum(power, budget, NOISE_VAR, hermitian=True)
    top_k = design.diagonal_bayes(
        greedy_a_optimal(power, budget), power, NOISE_VAR, hermitian=True
    )["bayes_mmse_total"]
    assert top_k > optimum * 1.5


def test_orbit_aware_greedy_is_exactly_optimal():
    """Separable and concave across orbits, so greedy increments are optimal."""
    for seed in (3, 11, 42):
        power = _symmetric_spectrum((4, 4), seed=seed)
        for budget in (4, 6, 9):
            optimum, _ = _exhaustive_optimum(power, budget, NOISE_VAR, hermitian=True)
            greedy = design.diagonal_bayes(
                greedy_a_optimal_hermitian(power, budget, noise_var=NOISE_VAR),
                power,
                NOISE_VAR,
                hermitian=True,
            )["bayes_mmse_total"]
            assert greedy == pytest.approx(optimum, rel=1e-12)


def test_orbit_aware_greedy_respects_the_budget_and_the_center():
    power = _symmetric_spectrum((8, 8))
    mask = greedy_a_optimal_hermitian(power, 20, noise_var=NOISE_VAR, n_center=4)
    assert int(mask.sum()) == 20
    assert set(np.unique(mask)) <= {0.0, 1.0}


def test_high_noise_makes_repeat_looks_worthwhile():
    """The selector must not simply forbid conjugate partners.

    The increment ratio between a second and a first look on the same orbit is
    ``sigma^2 / (sigma^2 + 2s)``, so a repeat on a dominant orbit beats a fresh
    look at a weak one once the noise dominates the prior. (2, 2) is the
    self-conjugate DC of a 4x4 centered grid and its orbit admits only one
    look, so the dominant orbit is placed off-center and mirrored, as a real
    signal's spectrum requires.
    """
    power = np.full((4, 4), 1.0)
    power[1, 1] = power[3, 3] = 500.0
    quiet = greedy_a_optimal_hermitian(power, 4, noise_var=1e-6)
    loud = greedy_a_optimal_hermitian(power, 4, noise_var=1e6)
    assert not np.array_equal(quiet, loud)
    # Negligible noise: one look on the dominant orbit is plenty.
    assert quiet[1, 1] + quiet[3, 3] == 1.0
    # Overwhelming noise: averaging both members of the dominant orbit wins.
    assert loud[1, 1] + loud[3, 3] == 2.0


def test_mutual_information_is_monotone_in_the_mask():
    power = _symmetric_spectrum((8, 8))
    rng = np.random.default_rng(0)
    small = np.zeros((8, 8))
    small.ravel()[rng.choice(64, 20, replace=False)] = 1.0
    bigger = small.copy()
    zeros = np.flatnonzero(bigger.ravel() == 0.0)
    bigger.ravel()[zeros[:10]] = 1.0
    for hermitian in (True, False):
        a = design.diagonal_bayes(small, power, NOISE_VAR, hermitian=hermitian)
        b = design.diagonal_bayes(bigger, power, NOISE_VAR, hermitian=hermitian)
        assert b["mutual_information_nats"] >= a["mutual_information_nats"]
        assert b["bayes_mmse_total"] <= a["bayes_mmse_total"]


def test_independent_model_matches_the_closed_form():
    power = _symmetric_spectrum((8, 8))
    rng = np.random.default_rng(1)
    mask = np.zeros((8, 8))
    mask.ravel()[rng.choice(64, 25, replace=False)] = 1.0
    got = design.diagonal_bayes(mask, power, NOISE_VAR, hermitian=False)
    expected = np.where(
        mask > 0.5, power * NOISE_VAR / (power + NOISE_VAR), power
    ).sum()
    assert got["bayes_mmse_total"] == pytest.approx(expected)
    assert got["bayes_mmse_per_pixel"] == pytest.approx(expected / 64)


def test_subspace_bayes_matches_a_direct_posterior():
    rng = np.random.default_rng(0)
    shape = (8, 8)
    d = 5
    basis, _ = np.linalg.qr(rng.standard_normal((64, d)))
    eigenvalues = np.linspace(1.0, 0.1, d)
    mask = np.zeros(shape)
    mask.ravel()[rng.choice(64, 24, replace=False)] = 1.0

    from mrsim.subspace import to_kspace_basis

    phi = to_kspace_basis(basis, shape)[np.flatnonzero(mask.ravel() > 0.5)]
    precision = phi.conj().T @ phi / NOISE_VAR + np.diag(1.0 / eigenvalues)
    expected = float(np.trace(np.linalg.inv(precision)).real)

    got = design.subspace_bayes(mask, basis, eigenvalues, NOISE_VAR)
    assert got["bayes_mmse_total"] == pytest.approx(expected)
    assert got["mutual_information_nats"] > 0.0


def test_design_rejects_bad_inputs():
    power = np.ones((4, 4))
    with pytest.raises(ValueError):
        design.diagonal_bayes(np.full((4, 4), 0.5), power, NOISE_VAR)
    with pytest.raises(ValueError):
        design.diagonal_bayes(np.ones((4, 4)), power, 0.0)
    with pytest.raises(ValueError):
        design.diagonal_bayes(np.ones((2, 2)), power, NOISE_VAR)
