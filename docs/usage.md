# Usage

Run scripts from the repository root with `--config <path>`. The default config
is `configs/default.yaml`. Each resolved config gets a stable fingerprint, and
outputs are written to:

```text
runs/<experiment_name>/<config-hash>/
```

Each script also saves a JSON manifest containing the full config, command-line
arguments, package versions, platform, and Git revision when available.

## Core scripts

| Script | Purpose |
| --- | --- |
| `01_generate_synthetic_data.py` | Generate the configured synthetic dataset and a preview. |
| `02_make_baseline_masks.py` | Build and evaluate the stable baseline masks. |
| `03_reconstruct_and_evaluate.py` | Evaluate saved masks, falling back to baselines when none are present. |
| `04_aopt_greedy_mask.py` | Run the opt-in diagonal-prior A-optimal design. |
| `05_artifact_aware_mask_search.py` | Run the opt-in PSF-penalized A-optimal design and optional beta sweep. |
| `06_greedy_data_driven_mask.py` | Run the opt-in train-spectrum greedy design. |
| `07_compare_all_masks.py` | Build exactly the masks in `mask.types`, reconstruct, summarize, and plot. |

The default evidence path is script 07 with the five non-learned baselines in
`configs/default.yaml`. Scripts 04-06 are available for targeted ablations but
are not enabled by default.

## Experimental scripts

| Script | Status and purpose |
| --- | --- |
| `08_subspace_mask.py` | Experimental linear-subspace design and reconstruction. |
| `09_loupe_baseline.py` | Experimental LOUPE-style learned Cartesian line mask and U-Net. |
| `10_compare_manifold_vs_learned.py` | Experimental comparison of subspace, diagonal-prior, and learned designs. |
| `11_budget_sweep.py` | Exploratory parameter and acceleration sweep; correlations are descriptive diagnostics only. |
| `12_train_unet.py` | Experimental U-Net post-processor trained on the synthetic train split. |

These scripts are not executed by CI and are not part of the default result
claim. Learned, subspace, and LOUPE-style results require independent seeds,
held-out model selection, and an explicitly defined shift protocol before they
can support comparative conclusions.

## Shipped configurations

### `configs/default.yaml`

- `data`: 60 images of size 64x64; 36 train, 12 validation, 12 test.
- `data.phantom`: independent draws from `ellipses` for every split.
- `measurement.noise_std`: 0.005 in complex Fourier space.
- `mask.sampling_fraction`: 0.25.
- `mask.types`: `uniform_random`, `variable_density`,
  `multilevel_random`, `equispaced_lines`, `variable_density_lines`.
- `recon.wavelet_ista`: 15 iterations, `db4`, 3 levels,
  `final_dc: false`.
- `outputs.n_examples`: 3.

The validation split is reserved between train and test. Script 07 estimates
the frequency-domain mean, centered variance, and second moment from the train
split and evaluates only the test split; it does not tune on the test data.

`final_dc` is false because the default measurement contains noise. Replacing
the measured coefficients exactly at the last iteration would also replace
them with their noisy values. Set it to true only when a hard final
data-consistency projection is the intended estimator.

### `configs/smoke.yaml`

- 8 images of size 32x32; 4 train, 2 validation, 2 test.
- 2 masks: `uniform_random` and `equispaced_lines`.
- 2 wavelet-ISTA iterations and one saved example.

This config verifies imports, data flow, file creation, and the main script. It
is too small for method ranking or scientific interpretation.

## Configuration keys

- `experiment_name`: first component of the run directory.
- `seed`: Python, NumPy, and PyTorch seed.
- `data.n_images`, `data.image_size`: generated dataset size and square image
  side length.
- `data.n_train`, `data.n_val`, `data.n_test`: deterministic contiguous split
  sizes. Their sum must not exceed `n_images`.
- `data.phantom`: `ellipses` or `shepp_logan`.
- `measurement.noise_std`: standard deviation of simulated circular complex
  Gaussian k-space noise.
- `mask.sampling_fraction`: fraction of Fourier locations acquired.
- `mask.center_fraction`: fraction of the point budget reserved near the
  Fourier center for point-mask generators.
- `mask.variable_density_decay`: point-wise variable-density exponent.
- `mask.lines.n_center_lines`, `mask.lines.decay`: settings for Cartesian
  column masks.
- `mask.multilevel.n_levels`, `mask.multilevel.decay`: annular multilevel
  baseline settings.
- `mask.types`: masks built by script 07.
- `recon.ridge_lambda`: scalar fallback used when no train spectrum is
  supplied.
- `recon.wiener_lambda` (optional): overrides the default Wiener weight
  `measurement.noise_std ** 2`.
- `recon.wavelet_ista`: `threshold`, `n_iters`, `wavelet`, `levels`, and
  `final_dc`. Remove this block to skip wavelet ISTA.
- `outputs.n_examples`: number of examples saved per mask and method.
- `experimental.include_unet_in_comparisons`: opt-in switch for loading a
  compatible checkpoint produced by script 12 into scripts 07 and 11.

Set this switch before running script 12, then use the same unchanged config
for comparison; the checkpoint and run directory are config-fingerprinted.

The `greedy`, `subspace`, `unet_post`, `budget_sweep`, and `loupe` blocks
configure opt-in experimental scripts. Their presence does not enable those
methods in script 07; only `mask.types` controls the default mask comparison.

## Acquisition strata

The code supports two mask geometries:

1. Point masks select individual 2-D Fourier coefficients:
   `uniform_random`, `variable_density`, and `multilevel_random`.
2. Cartesian line masks select columns:
   `equispaced_lines` and `variable_density_lines`.

At 64x64 and 25% sampling, the default budget is 1024 points, exactly 16 full
columns. At 32x32 and 25% sampling, the smoke budget is 256 points, exactly 8
full columns. For a non-divisible point budget, line designs leave the
remainder unspent; output tables record their actual sample count and
acceleration. Do not compare a point design against a line design as if they
had the same hardware constraints.

## Output layout

```text
runs/<experiment_name>/<config-hash>/
  config_<script>.json
  data/dataset.pt
  data/preview.png
  masks/<name>.npy
  masks/<name>.png
  masks/manifest.json
  psf/<name>_psf.png
  metrics/<prefix>_metrics.csv
  metrics/<prefix>_psf_metrics.csv
  metrics/summary.csv
  metrics/argumentation.csv
  metrics/argumentation_correlations.csv
  metrics/budget_sweep.csv
  metrics/budget_sweep_correlations.csv
  models/unet_post.pt
  recon/<mask>_<method>.png
  artifact_maps/<mask>_<method>.png
  artifact_maps/<mask>_<method>_artifact_field.png
  artifact_maps/<mask>_<method>_nullspace.png
  plots/score_vs_error_<acquisition_family>.png
  plots/psf_profiles.png
  plots/zoom_comparison.png
```

Some files are produced only by the corresponding experimental script.

## Interpretation classes

### Operator/measurement-observable

The mask, sample count, PSF, measured k-space values, and a reconstruction
residual `||y - A x_hat||` require no test ground truth. PSF metrics describe
the operator geometry; they do not certify reconstruction error by themselves.

### Prior-derived but test-truth-free

`mask_score`, `prior_observable_energy_fraction`, weighted PSF,
`wavelet_leakage`, and subspace leakage are estimated from training data or a
chosen representation. They are design-time diagnostics, but their validity
under distribution shift is an assumption to be tested rather than a
distribution-free guarantee.

### Oracle evaluation only

The following current outputs use the known synthetic test image:

- `complex_mse = mean(|x_hat-x|^2)`;
- `magnitude_mse = mean((|x_hat|-|x|)^2)`, PSNR, SSIM, and NRMSE;
- `oracle_unsampled_energy_ratio = ||(I-P)x||^2 / ||x||^2`;
- `oracle_nullspace_error_norm = ||(I-P)(x_hat-x)||`;
- `oracle_observed_subspace_error_norm = ||P(x_hat-x)||`;
- `oracle_truth_nullspace_norm = ||(I-P)x||`.

They are valid offline evaluation metrics in simulation. They are not
available for an unknown deployed sample and must not be presented as
observable per-sample certificates.

## Correlation outputs

Script 07 and the exploratory sweep write descriptive Spearman correlations
within acquisition strata. They intentionally omit nominal p-values because
the constructed masks are dependent, selected designs rather than iid
observations. The repository does not implement repeated-seed inference or an
independent confirmatory dataset.
