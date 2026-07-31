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

python scripts/13_axis_precheck.py --config configs/default.yaml

python scripts/11_budget_sweep.py --config configs/default.yaml
```

The shipped default uses 400 synthetic 64x64 images split into 200 train, 100
validation, and 100 test images. It compares six baselines: `uniform_random`,
`variable_density`, `multilevel_random`, `multilevel_local_coherence`,
`equispaced_lines`, and `variable_density_lines`.

Summary tables carry percentile bootstrap intervals over images. Several mask
differences in the default family are smaller than those intervals, so read the
intervals rather than the means.

For per-sample error certificates:

```bash
python scripts/14_conformal_certificate.py --config configs/default.yaml
```

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

## Real signals: nominal versus effective budget

The synthetic signals are real, so `X(-k) = conj(X(k))` and sampling both
members of a conjugate pair acquires one complex unknown rather than two. The
effective budget is the number of conjugate orbits a mask touches, reported as
`effective_samples` and `hermitian_redundancy`.

At the shipped 25% budget these diverge sharply: `uniform_random` reaches 875
effective samples while `equispaced_lines` reaches 514, because its column set
is closed under conjugation and half its acquisition is redundant. Any
"equal-budget" statement should name which budget it means.

This also removes the usual justification for top-k mask design. A real
signal's power spectrum is symmetric, so ranking by prior power selects both
members of each pair; `greedy.greedy_a_optimal_hermitian` allocates over orbits
instead and halves the exact Bayes MMSE at the default settings.

## Exact design quantities

`mrsim/design.py` computes the Bayes MMSE and mutual information in closed form
for the diagonal spectral prior and the PCA subspace prior, plus an optimality
gap in nats against the exact binary optimum. These are the quantities that
`rho`, `wavelet_leakage`, and `mask_score` were approximating.

`mrsim/certify.py` provides split-conformal error certificates with
finite-sample marginal coverage under exchangeability. See
[`docs/experiments.md`](docs/experiments.md) for what that does and does not
guarantee — in particular it is not conditional coverage, which is provably
unattainable distribution-free.

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
scripts/            core scripts 01-07 and experimental scripts 08-13
tests/              unit tests
docs/               usage, interpretation, and reproduction guidance
runs/               generated outputs (gitignored)
```

Scripts 08-13 cover subspace, LOUPE-style, learned post-processing, broad
mask-family sweeps, and design-time diagnostics. They are experimental opt-in
studies, are not part of the default evidence, and require separate validation
before being used for paper claims.

## Design-time coverage and recoverability diagnostics

`scripts/13_axis_precheck.py` (with per-budget variants from
`scripts/11_budget_sweep.py`) separates the two prior-derived questions a mask
design must answer:

- **Coverage** — how much representation energy the mask observes:
  radial sampling density against a decay-2 variable-density reference, the
  radial profile of the observed-energy fraction, and a per-subband wavelet
  leakage heatmap whose energy-weighted rows reproduce the scalar
  `wavelet_leakage` metric exactly.
- **Recoverability** — how well-conditioned the observed content is: the full
  singular spectrum of the restricted operator (never just its minimum;
  rank-deficient masks are excluded from scalar comparisons), and a per-subband
  sigma_min profile in which orientation bands expose direction and the level
  hierarchy exposes scale — e.g. Cartesian column masks collapse one
  orientation per scale while leaving the orthogonal orientation conditioned.

  Conditioning is measured over a *distribution* of sparse supports, not one
  aggregated support. That distinction decides the answer: against a single
  fixed support, `sigma_min` correlates with coverage at +0.963 and looks
  redundant; against 16 drawn supports the correlation falls to +0.477
  (energy-weighted) or +0.387 (tree-structured), so it does carry information
  coverage does not.

These are prior-derived design diagnostics in the sense of the taxonomy above:
computable before test evaluation, but dependent on the training distribution
and the chosen representation.
