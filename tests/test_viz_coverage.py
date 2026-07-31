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


def test_short_names_compress_family_members():
    from mrsim.viz import short_name

    assert short_name("variable_density_d2.5_s0") == "vd2.5s0"
    assert short_name("psf_penalized_b4") == "psf-b4"
    assert short_name("multilevel_L4_d1.5") == "ml-L4d1.5"
    assert short_name("variable_density_lines_d1") == "vdl1"
    assert short_name("uniform_random_s2") == "uni-s2"
    assert short_name("aopt_greedy") == "aopt"
    assert short_name("unknown_mask") == "unknown_mask"
    # Distinct names must stay distinct after compression.
    names = ["variable_density_d2.5_s0", "variable_density_d2.5_s1",
             "psf_penalized_b1", "psf_penalized_b4", "multilevel_L3_d2"]
    assert len({short_name(n) for n in names}) == len(names)


def test_design_family_covers_the_whole_default_family():
    from mrsim.viz import design_family

    expected = {
        "uniform_random_s0": "uniform",
        "variable_density_d3.5_s1": "vd",
        "variable_density_lines_d2": "lines",
        "multilevel_L4_d1.5": "multilevel",
        "equispaced_lines": "lines",
        "line_aopt": "lines",
        "spectrum_energy_greedy": "lines",
        "loupe_learned": "lines",
        "aopt_greedy": "psf",
        "psf_penalized_b2": "psf",
        "subspace_aopt_b0": "subspace",
        "something_else": "other",
    }
    for name, family in expected.items():
        assert design_family(name) == family, name


def test_scatter_with_labels_handles_full_family(tmp_path):
    from mrsim.viz import scatter_with_labels

    rng = np.random.default_rng(0)
    names = (
        [f"uniform_random_s{i}" for i in range(5)]
        + [f"variable_density_d{d}_s0" for d in (1.5, 2.5, 3.5, 5.0)]
        + [f"multilevel_L{L}_d1.5" for L in (3, 4, 5)]
        + [f"psf_penalized_b{b}" for b in (0.5, 1, 2, 4)]
        + ["equispaced_lines", "line_aopt", "subspace_aopt_b0", "aopt_greedy"]
    )
    out = tmp_path / "scatter.png"
    scatter_with_labels(
        rng.random(len(names)), rng.random(len(names)), names, out,
        xlabel="x", ylabel="y", always_label=["aopt_greedy"],
    )
    assert out.exists()
