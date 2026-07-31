"""Conjugate-orbit accounting for real-valued signals.

The synthetic signals in this repository are real, so ``X(-k) = conj(X(k))``
and a mask that samples both members of a conjugate pair acquires one complex
unknown rather than two. These tests pin the effective-budget numbers for the
shipped default configuration, because the nominal equal-budget comparison
hides a large spread in them.
"""

from __future__ import annotations

import numpy as np
import pytest

from mrsim import artifacts, masks
from mrsim.fft_ops import fft2c

import torch

SHAPE = (64, 64)
N_SAMPLES = 1024  # 25% of 64x64, the shipped default budget


def test_partner_map_is_an_involution_for_both_parities():
    for shape in [(64, 64), (32, 16), (5, 7), (8, 5)]:
        rows, cols = artifacts.conjugate_partner_index(shape)
        assert np.array_equal(rows[rows], np.arange(shape[0]))
        assert np.array_equal(cols[cols], np.arange(shape[1]))


def test_partner_map_matches_conjugate_symmetry_of_a_real_image():
    rng = np.random.default_rng(0)
    image = rng.standard_normal((16, 16))
    spectrum = fft2c(torch.from_numpy(image).to(torch.complex128)).numpy()
    assert np.allclose(artifacts.conjugate_reflect(spectrum), spectrum.conj())


def test_full_mask_has_no_redundancy_beyond_the_pairing():
    full = np.ones(SHAPE, dtype=np.float32)
    # Every orbit is covered. A 64x64 centered grid has exactly four
    # self-conjugate frequencies (DC and Nyquist on each axis), which are the
    # only orbits of size one, so the count sits just above half the grid and
    # the redundancy just below 1/2.
    assert artifacts.effective_sample_count(full) == (4096 + 4) // 2 == 2050
    assert artifacts.hermitian_redundancy(full) == pytest.approx(1 - 2050 / 4096)
    assert artifacts.hermitian_redundancy(full) < 0.5


def test_equispaced_lines_are_conjugation_closed():
    """The default line mask spends exactly half its budget on known values."""
    mask = masks.equispaced_lines_mask(SHAPE, N_SAMPLES)
    coverage = artifacts.conjugate_orbit_coverage(mask)
    # Closed under conjugation: reflecting the mask changes nothing.
    assert np.array_equal(coverage, mask.astype(np.float64))
    assert artifacts.effective_sample_count(mask) == 514
    assert artifacts.hermitian_redundancy(mask) == pytest.approx(0.498, abs=5e-4)


def test_default_family_effective_budgets_are_far_apart():
    """Equal nominal budgets hide a 1.75x spread in effective complex DOF."""
    rng = np.random.default_rng(0)
    family = {
        "uniform_random": masks.uniform_random_mask(SHAPE, N_SAMPLES, rng),
        "variable_density": masks.variable_density_mask(SHAPE, N_SAMPLES, rng, decay=3.0),
        "multilevel_random": masks.multilevel_random_mask(
            SHAPE, N_SAMPLES, rng, n_levels=4, decay=1.5
        ),
        "equispaced_lines": masks.equispaced_lines_mask(SHAPE, N_SAMPLES),
        "variable_density_lines": masks.variable_density_lines_mask(
            SHAPE, N_SAMPLES, rng, decay=2.0
        ),
    }
    effective = {name: artifacts.effective_sample_count(m) for name, m in family.items()}
    for name, mask in family.items():
        assert int(mask.sum()) == N_SAMPLES, name
        assert 1 <= effective[name] <= N_SAMPLES

    # Measured on the shipped default configuration at seed 0.
    assert effective["uniform_random"] == 901
    assert effective["variable_density_lines"] == 865
    assert effective["variable_density"] == 758
    assert effective["multilevel_random"] == 750
    assert effective["equispaced_lines"] == 514

    spread = max(effective.values()) / min(effective.values())
    assert spread > 1.7, "the equal-budget comparison is confounded; keep reporting it"


def test_orbit_energy_fraction_is_at_least_the_pointwise_one():
    rng = np.random.default_rng(1)
    power = rng.random(SHAPE) ** 2
    power = power + artifacts.conjugate_reflect(power)  # a real signal's spectrum
    for mask in [
        masks.uniform_random_mask(SHAPE, N_SAMPLES, rng),
        masks.equispaced_lines_mask(SHAPE, N_SAMPLES),
    ]:
        pointwise = artifacts.prior_observable_energy_fraction(mask, power)
        orbit = artifacts.prior_observable_energy_fraction_real(mask, power)
        assert orbit >= pointwise - 1e-12
        assert 0.0 <= orbit <= 1.0 + 1e-12


def test_orbit_energy_fraction_rejects_bad_inputs():
    power = np.ones(SHAPE)
    with pytest.raises(ValueError):
        artifacts.prior_observable_energy_fraction_real(np.full(SHAPE, 0.5), power)
    with pytest.raises(ValueError):
        artifacts.prior_observable_energy_fraction_real(np.ones((8, 8)), power)
