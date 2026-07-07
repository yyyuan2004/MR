# Usage

All scripts run from the repository root, take `--config <path>` (default
`configs/default.yaml`), and write to `runs/<experiment_name>/`. Every script
saves a JSON snapshot of the config it ran with
(`runs/<exp>/config_<script>.json`).

## Scripts

| Script | Purpose |
| --- | --- |
| `01_generate_synthetic_data.py` | Generate and save the dataset plus a preview grid. |
| `02_make_baseline_masks.py` | Build uniform random, variable density, and equispaced line masks; evaluate on the test split. |
| `03_reconstruct_and_evaluate.py` | Reconstruct the test split with every mask found in `runs/<exp>/masks/*.npy` (falls back to baselines). |
| `04_aopt_greedy_mask.py` | Greedy Bayesian A-optimal mask from a power-law prior fitted on the train split. |
| `05_artifact_aware_mask_search.py` | Greedy mask trading expected MSE gain against the PSF max sidelobe. |
| `06_greedy_data_driven_mask.py` | Greedy mask from the empirical mean spectral energy of the train split. |
| `07_compare_all_masks.py` | Build every mask in `mask.types`, evaluate, summarize, and plot score vs error. |

Scripts 02-07 generate the dataset automatically if `runs/<exp>/data/dataset.pt`
does not exist, so each script is runnable on its own.

## Config reference (`configs/default.yaml`)

- `experiment_name`: output directory name under `runs/`.
- `seed`: global seed (Python, NumPy, PyTorch).
- `data.n_images`, `data.image_size`: dataset size and image side length.
- `data.n_train`, `data.n_test`: deterministic split (first `n_train` images
  train, next `n_test` test).
- `data.phantom`: `ellipses` (random ellipse superpositions) or `shepp_logan`.
- `measurement.noise_std`: std of complex Gaussian frequency-domain noise.
  Single source of truth: the greedy criterion's noise variance and the
  Wiener regularization weight both default to `noise_std ** 2`.
- `mask.sampling_fraction`: fraction of frequency-domain locations measured;
  the measurement budget is `round(fraction * H * W)` and is met exactly by
  every generator.
- `mask.center_fraction`: fraction of the budget forced onto the lowest
  spatial frequencies.
- `mask.variable_density_decay`: polynomial decay exponent of the variable
  density profile.
- `mask.types`: masks compared by script 07. Valid names: `uniform_random`,
  `variable_density`, `equispaced_lines`, `aopt_greedy`,
  `artifact_aware_greedy`, `data_driven_greedy`.
- `greedy.noise_var`: noise variance in the A-optimal gain
  `s_k^2 / (s_k + noise_var)`. Defaults to `measurement.noise_std ** 2`; set
  it only to deliberately override that invariant.
- `greedy.artifact_beta`: weight of the PSF max-sidelobe penalty in the
  artifact-aware score.
- `greedy.n_candidates`: candidate locations evaluated per artifact-aware step.
- `recon.ridge_lambda`: scalar shrinkage weight, used only on the fallback
  path when no spectrum is available.
- `recon.wiener_lambda` (optional): Wiener regularization weight; defaults to
  `measurement.noise_std ** 2`.
- `recon.wavelet_ista`: iterative soft-thresholding parameters (`threshold`,
  `n_iters`, `wavelet`, `levels`, `final_dc`). Remove the block to skip the
  method.
- `outputs.n_examples`: number of representative examples saved as image grids.

## Output layout

```
runs/<experiment_name>/
  config_<script>.json      config snapshot per script
  data/dataset.pt           image stack (torch tensor, N x H x W)
  data/preview.png          first 16 images
  masks/<name>.npy          binary mask arrays
  masks/<name>.png          mask images
  psf/<name>_psf.png        mask, log-magnitude PSF, center-row profile
  metrics/<prefix>_metrics.csv       per-image metrics
  metrics/<prefix>_psf_metrics.csv   PSF metrics per mask
  metrics/summary.csv                aggregated comparison (script 07)
  recon/<mask>_<method>.png          reconstruction grids
  artifact_maps/<mask>_<method>.png  total error |recon - truth| grids
  artifact_maps/<mask>_<method>_artifact_field.png  null-space error |(I-P)(recon-truth)|
  artifact_maps/<mask>_<method>_nullspace.png       invented content |(I-P) recon|
  plots/score_vs_error.png           mask score vs measured error (script 07)
```

## Metrics

- `mse`, `psnr`, `ssim`, `nrmse`: computed on magnitude reconstructions against
  the ground-truth image.
- `aliasing_energy_ratio`: `||(I - P) x||^2 / ||x||^2`, the fraction of image
  energy lost by the sampling projector `P = F^H M F`.
- `psf_max_sidelobe`: largest off-peak PSF magnitude relative to the peak
  (mask coherence).
- `psf_sidelobe_energy`: off-peak fraction of PSF energy (depends mostly on the
  budget; reported for completeness).
- `mask_score`: expected zero-filled per-pixel MSE under the train mean power
  spectrum (unmeasured spectral energy / number of pixels).
- `artifact_norm`, `consistency_norm`, `recon_nullspace_norm`,
  `truth_nullspace_norm`, `no_nullspace_content`: per-image error
  decomposition by the orthogonal projector `P = F^H M F` — null-space
  imputation error, observed-subspace error, the reconstruction's and the
  reference signal's null-space content, and a norm-based flag that is True
  for reconstructions confined to the observed subspace. See
  `docs/experiments.md`.
