[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/yyyuan2004/MR)

# MR

Small, CPU-oriented experiments for comparing undersampled Fourier masks and
reconstruction errors. The default path generates synthetic images, applies an
ideal single-coil Cartesian Fourier operator, reconstructs with fixed classical
methods, and records an observed-subspace / null-space error decomposition.

## Scope

The checked-in default experiment is deliberately narrow:

- train, validation, and test images are independent draws from the same
  synthetic ellipse generator;
- the forward model is an ideal centered orthonormal 2-D Fourier transform with
  a binary mask and circular complex Gaussian k-space noise;
- there are no coil sensitivities, non-Cartesian trajectories, calibration
  errors, real scanner data, CT data, or distribution shift;
- the default comparison contains only stable, non-learned point and Cartesian
  line baselines.

Results from this configuration are synthetic, in-distribution sanity checks.
They do not establish clinical performance, robustness under prior shift,
regime-transition claims, or confirmatory inference.

## Installation

Python 3.11 or newer is required. CPU-only PyTorch is sufficient:

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev]"
```

`requirements.txt` contains runtime dependencies only. The editable install
with the `dev` extra also installs pytest.

## Quickstart

Run the default CPU experiment:

```bash
python scripts/07_compare_all_masks.py --config configs/default.yaml
```

The shipped default uses 60 synthetic 64x64 images split into 36 train, 12
validation, and 12 test images. It compares five baselines:
`uniform_random`, `variable_density`, `multilevel_random`,
`equispaced_lines`, and `variable_density_lines`.

For a minimal wiring check, use:

```bash
python scripts/07_compare_all_masks.py --config configs/smoke.yaml
```

The smoke config uses eight 32x32 images and two masks. It is intended for CI,
not for scientific conclusions.

Run unit tests with:

```bash
pytest -q
```

GitHub Actions runs the tests and the smoke experiment on Ubuntu with Python
3.11 and CPU PyTorch.

## Reading the reported quantities

The output mixes quantities with different information requirements. Keep
these categories separate:

- **Operator/measurement-observable:** the mask, sampling budget, PSF, measured
  data, and a measurement residual computed from a reconstruction.
- **Prior-derived design diagnostics:** `mask_score`,
  `prior_observable_energy_fraction`, `weighted_max_sidelobe`, and
  `wavelet_leakage`. They are computable before seeing test truth, but depend
  on the training distribution or a chosen representation and are not
  distribution-free certificates.
- **Oracle evaluation quantities:** `complex_mse`, `magnitude_mse`, PSNR,
  SSIM, NRMSE, `oracle_unsampled_energy_ratio`,
  `oracle_nullspace_error_norm`,
  `oracle_observed_subspace_error_norm`, and
  `oracle_truth_nullspace_norm`. These use the ground-truth test image, so
  they are unavailable for certifying an unknown test sample.

`measurement_residual_norm = ||y - A recon||` is reported separately and is
directly computable from a measurement and reconstruction. It is not the same
as the oracle observed-subspace error.

## Point masks and line masks

Point masks select arbitrary individual Fourier coefficients. Line masks
select Cartesian columns. These are different acquisition constraints and
should be summarized in separate strata rather than treated as interchangeable
designs.

Every line-mask helper emits only complete columns. For a point budget that is
not divisible by image height, the remainder is left unspent and the manifest
records the actual sample count.

## Output layout

Every config is fingerprinted. Outputs are written under:

```text
runs/<experiment_name>/<config-hash>/
```

Typical contents include config snapshots, masks, PSFs, per-image metrics,
acquisition-stratified summary plots, reconstruction grids, and oracle error maps. See
[`docs/usage.md`](docs/usage.md) for the complete layout and configuration
reference.

## Repository structure

```text
configs/            default and CI smoke configurations
mrsim/              simulation, masks, reconstruction, metrics, and plotting
scripts/            core scripts 01-07 and experimental scripts 08-12
tests/              unit tests
docs/               usage, interpretation, and reproduction guidance
runs/               generated outputs (gitignored)
```

Scripts 08-12 cover subspace, LOUPE-style, learned post-processing, and broad
mask-family sweeps. They are experimental opt-in studies, are not part of the
default evidence, and require separate validation before being used for paper
claims.
