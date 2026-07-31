"""Why the coverage metrics agree, stated precisely.

``docs/experiments.md`` reports a Spearman correlation of -0.994 between
``rho`` and ``wavelet_leakage`` and concludes the two are interchangeable. The
conclusion holds for the shipped mask family, but the reason matters, because
it decides whether the finding generalizes.

Two separate facts, both checked here:

1. Both metrics are *exactly affine* in the mask, ``c - <M, w>``. This is an
   algebraic identity, not an empirical observation, and it holds to machine
   precision.
2. Being affine is nevertheless **not** enough to force rank collapse. The two
   weight vectors have a cosine similarity of only about 0.30, and over
   unstructured masks at the same budget the rank correlation is around -0.36.
   The collapse comes from the mask *family*: every member is parameterized by
   how tightly it concentrates toward low frequency, and both functionals track
   that single latent axis at |rho| ~ 0.97.

So the interchangeability is a property of the family, not of the metrics. A
family that varied something other than radial concentration could separate
them, and any future claim of redundancy has to name the family it holds over.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import spearmanr

from mrsim import artifacts, masks

SHAPE = (64, 64)
BUDGET = 1024


@pytest.fixture(scope="module")
def wavelet_weights():
    rng = np.random.default_rng(0)
    train = np.stack(
        [
            masks.radius_map(SHAPE) * 0.0
            + rng.random(SHAPE)  # cheap stand-in stack; only its statistics matter
            for _ in range(8)
        ]
    )
    mass = artifacts.subband_spectral_mass(SHAPE, wavelet="db4", levels=3)
    energies = artifacts.subband_energies(train, wavelet="db4", levels=3)
    weights = sum(energies[key] * mass[key] for key in mass) / sum(energies.values())
    return mass, energies, weights


def test_wavelet_leakage_is_exactly_affine_in_the_mask(wavelet_weights):
    """Each subband's spectral mass sums to one, so leakage = 1 - <M, w>."""
    mass, energies, weights = wavelet_weights
    rng = np.random.default_rng(1)
    for _ in range(5):
        flat = np.zeros(SHAPE[0] * SHAPE[1])
        flat[rng.choice(flat.size, BUDGET, replace=False)] = 1.0
        mask = flat.reshape(SHAPE)
        measured = artifacts.wavelet_leakage_score(mask, mass, energies)
        affine = 1.0 - float((mask * weights).sum())
        assert measured == pytest.approx(affine, abs=1e-12)


def test_rho_is_exactly_affine_in_the_mask():
    rng = np.random.default_rng(2)
    power = rng.random(SHAPE) + 0.05
    weights = power / power.sum()
    flat = np.zeros(SHAPE[0] * SHAPE[1])
    flat[rng.choice(flat.size, BUDGET, replace=False)] = 1.0
    mask = flat.reshape(SHAPE)
    measured = artifacts.prior_observable_energy_fraction(mask, power)
    assert measured == pytest.approx(float((mask * weights).sum()), abs=1e-12)


def test_affine_alone_does_not_collapse_the_ranking(wavelet_weights):
    """The key negative control: unstructured masks do not show the collapse."""
    _, _, leak_weights = wavelet_weights
    rng = np.random.default_rng(3)
    power = np.exp(-masks.radius_map(SHAPE) / 6.0) + 1e-3
    rho_weights = power / power.sum()

    family = np.zeros((150, SHAPE[0] * SHAPE[1]))
    for row in family:
        row[rng.choice(row.size, BUDGET, replace=False)] = 1.0

    rho = family @ rho_weights.ravel()
    leakage = 1.0 - family @ leak_weights.ravel()
    correlation = spearmanr(rho, leakage).statistic
    assert abs(correlation) < 0.8, (
        "two affine functionals of the mask are not automatically redundant; "
        "if this fails the negative control is gone and the docs claim needs rework"
    )


def test_a_radially_parameterized_family_does_collapse(wavelet_weights):
    """One latent axis is what makes the two metrics agree."""
    _, _, leak_weights = wavelet_weights
    power = np.exp(-masks.radius_map(SHAPE) / 6.0) + 1e-3
    rho_weights = power / power.sum()

    rows = []
    for decay in np.linspace(0.5, 6.0, 12):
        mask = masks.variable_density_mask(
            SHAPE, BUDGET, np.random.default_rng(0), decay=float(decay)
        )
        rows.append(mask.ravel())
    family = np.stack(rows)

    rho = family @ rho_weights.ravel()
    leakage = 1.0 - family @ leak_weights.ravel()
    assert abs(spearmanr(rho, leakage).statistic) > 0.9
