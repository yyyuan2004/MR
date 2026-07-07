# Experiments

## Default experiment

`configs/default.yaml` defines the default experiment:

- 200 synthetic 64x64 random-ellipse phantoms; first 120 train, next 80 test.
- Sample budget: 25% of k-space (1024 of 4096 locations), with 2% of the
  budget forced onto the k-space center.
- Five masks: `uniform_random`, `variable_density`, `equispaced_lines`,
  `aopt_greedy`, `artifact_aware_greedy`.
- Complex Gaussian k-space noise with std 0.005.
- Reconstruction: zero-filled (`F^H y`) and ridge (per-coefficient shrinkage
  `1 / (1 + lambda)` on sampled locations).
- Outputs: per-image metrics CSV, aggregated summary CSV, mask/PSF images,
  reconstruction and artifact-map grids for 5 representative test images,
  and a scatter plot of mask score vs measured reconstruction error.

Run it with:

```bash
python scripts/07_compare_all_masks.py --config configs/default.yaml
```

## Mask types

- **uniform_random** — budget locations drawn uniformly without replacement.
  Low PSF coherence, but wastes samples on low-energy high frequencies.
- **variable_density** — sampling probability decays polynomially with |k|,
  concentrating samples where spectral energy is high while keeping the
  incoherence of random sampling.
- **equispaced_lines** — fully sampled columns on a regular grid (plus a
  partial column to meet the budget exactly). Highly coherent: the PSF has
  replica peaks, producing structured fold-over artifacts.
- **aopt_greedy** — greedy Bayesian A-optimal selection under a diagonal
  Gaussian prior in k-space. The prior spectrum is a radial power law fitted
  to the train split. Adding location k reduces the expected posterior MSE by
  `s_k^2 / (s_k + noise_var)`; the greedy loop always adds the largest
  remaining gain.
- **artifact_aware_greedy** — same gain, penalized by the PSF max sidelobe of
  the candidate mask: `score = gain / max_gain - beta * max_sidelobe`. This
  trades raw expected MSE against coherent aliasing structure.
- **data_driven_greedy** — uses the empirical mean spectral energy of the
  train images instead of a fitted prior. By Parseval, zero-filled MSE equals
  the unsampled spectral energy, so each greedy step adds the unsampled
  location with the largest measured mean `|X_k|^2`.

To include the data-driven mask in the comparison, add `data_driven_greedy`
to `mask.types` in the config.

## Mask score vs true error

The mask score is the expected zero-filled per-pixel MSE under the train mean
power spectrum. Script 07 plots this predicted score against the measured mean
MSE on the test split (`plots/score_vs_error.png`). Points near the diagonal
indicate the spectral model transfers from train to test; deviations flag
distribution shift, noise effects, or method-specific behavior (the ridge
points shift by the shrinkage factor).

## Variations

- Change `mask.sampling_fraction` to sweep the budget.
- Set `data.phantom: shepp_logan` for structured phantoms.
- Increase `measurement.noise_std` to see ridge separate from zero-filled.
- Raise `greedy.artifact_beta` to push the artifact-aware mask toward less
  coherent patterns; set it to 0 to recover plain A-optimal selection.
- Use a new `experiment_name` per variation so outputs land in separate
  `runs/<name>/` directories.
