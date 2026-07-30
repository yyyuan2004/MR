"""Frequency-coverage and wavelet-energy-coverage diagnostics.

These check the *data* behind the plots, not just that a PNG appears: the
radial profile must account for every sampled location, and the per-subband
leakage heatmap must reproduce artifacts.wavelet_leakage_score exactly when
its rows are energy-weighted.
"""

import numpy as np
import pytest

from mrsim.artifacts import subband_energies, subband_spectral_mass, wavelet_leakage_score
from mrsim.data import generate_dataset
from mrsim.masks import uniform_random_mask, variable_density_mask
from mrsim.viz import (
    plot_radial_coverage,
    plot_singular_spectra,
    plot_subband_leakage,
    representative_subset,
)

SIZE = 16


@pytest.fixture(scope="module")
def train_power() -> np.ndarray:
    from mrsim.experiment import mean_power_spectrum

    return mean_power_spectrum(generate_dataset(6, SIZE, seed=0))


@pytest.fixture(scope="module")
def mask_dict() -> dict:
    return {
        "uniform": uniform_random_mask((SIZE, SIZE), 64, np.random.default_rng(0)),
        "vd_shallow": variable_density_mask((SIZE, SIZE), 64, np.random.default_rng(1), decay=1.0),
        "vd_steep": variable_density_mask((SIZE, SIZE), 64, np.random.default_rng(2), decay=5.0),
    }


def test_radial_coverage_accounts_for_every_sample(tmp_path, mask_dict, train_power):
    out = tmp_path / "radial.png"
    table = plot_radial_coverage(mask_dict, train_power, out)
    assert out.exists()
    for name, mask in mask_dict.items():
        sampled = table[table["mask"] == name]["sampled"].sum()
        assert sampled == pytest.approx(float(mask.sum())), name
    # Empty radius bins are NaN by design (nothing to report there).
    assert table["rho_profile"].dropna().between(-1e-9, 1.0 + 1e-9).all()


def test_radial_profile_orders_masks_by_concentration(tmp_path, mask_dict, train_power):
    table = plot_radial_coverage(mask_dict, train_power, tmp_path / "radial.png")
    # A steeper density must observe more of the innermost bin's energy.
    inner = table.groupby("mask").first()["rho_profile"]
    assert inner["vd_steep"] >= inner["uniform"]


def test_subband_leakage_matches_scalar_score(tmp_path, mask_dict, train_power):
    images = generate_dataset(6, SIZE, seed=0).numpy()
    mass = subband_spectral_mass((SIZE, SIZE), wavelet="db2", levels=2)
    energies = subband_energies(images, wavelet="db2", levels=2)
    out = tmp_path / "subband.png"
    table = plot_subband_leakage(mask_dict, mass, energies, out)
    assert out.exists()
    # Energy-weighted row total == the repository's scalar leakage metric.
    for _, row in table.iterrows():
        expected = wavelet_leakage_score(mask_dict[row["mask"]], mass, energies)
        assert row["weighted_total"] == pytest.approx(expected, rel=1e-10)
    band_columns = [c for c in table.columns if c not in ("mask", "weighted_total")]
    assert ((table[band_columns] >= -1e-12) & (table[band_columns] <= 1.0 + 1e-12)).all().all()


def test_subband_sigma_profile_handles_zeros_and_missing_bands(tmp_path):
    from mrsim.viz import plot_subband_sigma_min

    out = tmp_path / "profile.png"
    plot_subband_sigma_min(
        {
            "point_mask": {"approx": 0.9, "level1_horizontal": 0.4, "level1_vertical": 0.5},
            "line_mask": {"approx": 0.8, "level1_horizontal": 0.0,  # dead orientation
                          "level1_vertical": 0.6},
        },
        out,
    )
    assert out.exists()


def test_singular_spectra_handles_exact_zeros(tmp_path):
    out = tmp_path / "spectra.png"
    plot_singular_spectra(
        {
            "full_rank": np.array([3.0, 1.0, 0.5]),
            "deficient": np.array([2.0, 1e-3, 0.0]),  # exact zero must not crash log scale
        },
        out,
    )
    assert out.exists()


def test_representative_subset_keeps_extremes(train_power):
    rng = np.random.default_rng(3)
    family = {
        f"m{i}": uniform_random_mask((SIZE, SIZE), 16 + 12 * i, rng) for i in range(12)
    }
    subset = representative_subset(family, train_power, n=5)
    assert len(subset) == 5
    total = float(train_power.sum())

    def rho(mask):
        return float((mask * train_power).sum()) / total

    rhos = sorted(rho(m) for m in family.values())
    kept = sorted(rho(m) for m in subset.values())
    assert kept[0] == pytest.approx(rhos[0])
    assert kept[-1] == pytest.approx(rhos[-1])


def test_representative_subset_is_identity_when_small(mask_dict, train_power):
    assert representative_subset(mask_dict, train_power, n=8) == mask_dict
