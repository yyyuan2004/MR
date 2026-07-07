# Experiments

## Default experiment

`configs/default.yaml` defines the default experiment:

- 200 synthetic 64x64 random-ellipse test signals; first 120 train, next 80 test.
- Measurement budget: 25% of the frequency domain (1024 of 4096 locations),
  with 2% of the budget forced onto the frequency-domain center.
- Five masks: `uniform_random`, `variable_density`, `equispaced_lines`,
  `aopt_greedy`, `artifact_aware_greedy`.
- Complex Gaussian frequency-domain noise with std 0.005. The same noise
  level drives the greedy criterion's noise variance and the Wiener
  regularization weight (single source of truth).
- Reconstruction: zero-filled (`F^H y`), Wiener (per-coefficient shrinkage
  `s_k / (s_k + noise_var)` with the spectrum estimated on the train split
  only), and wavelet ISTA (iterative soft-thresholding in a wavelet basis
  with a final data-consistency step).
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
  replica peaks, producing structured replica-aliasing artifacts.
- **aopt_greedy** — greedy Bayesian A-optimal selection under a diagonal
  Gaussian prior in the frequency domain. The prior spectrum is a radial power law fitted
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

## Error decomposition

The forward operator observes only part of the frequency domain, so signal
space splits into an observed subspace (the range of the orthogonal projector
`P = F^H M F`) and its null space. The pipeline reports three error
quantities per reconstruction:

- **Total error** `|recon - truth|`: everything, undifferentiated.
- **Observed-subspace (consistency) error** `P (recon - truth)`: disagreement
  with the measurements inside the observed subspace. For noiseless data and
  a data-consistent reconstruction this is ~0.
- **Null-space imputation error** `(I - P)(recon - truth)` (the *artifact
  field*): content the reconstruction invented or failed to restore in the
  unobserved directions. Its two ingredients are also reported separately:
  `(I - P) recon` (invented null-space content) and `(I - P) truth` (the
  reference signal's null-space component).

Linear diagonal reconstructions (zero-filling, Wiener) cannot place energy in
the null space, so their `recon_nullspace_norm` is ~0 and their artifact field
equals minus the reference null-space component. Nonlinear methods such as
wavelet ISTA impute null-space content; whether that imputation is faithful or
spurious is exactly what the per-image columns (`artifact_norm`,
`consistency_norm`, `recon_nullspace_norm`, `truth_nullspace_norm`,
`no_nullspace_content`) and the `_artifact_field.png` / `_nullspace.png`
magnitude maps make visible.

## Mask score vs true error

The mask score is the expected zero-filled per-pixel MSE under the train mean
power spectrum. Script 07 plots this predicted score against the measured mean
MSE on the test split (`plots/score_vs_error.png`). Points near the diagonal
indicate the spectral model transfers from train to test; deviations flag
distribution shift, noise effects, or method-specific behavior (Wiener
shrinkage, null-space imputation by wavelet ISTA).

## Variations

- Change `mask.sampling_fraction` to sweep the budget.
- Set `data.phantom: shepp_logan` for the structured Shepp-Logan test image.
- Increase `measurement.noise_std` to widen the gap between Wiener and
  zero-filling (the Wiener shrinkage scales with the noise variance).
- Raise `greedy.artifact_beta` to push the artifact-aware mask toward less
  coherent patterns; set it to 0 to recover plain A-optimal selection.
- Use a new `experiment_name` per variation so outputs land in separate
  `runs/<name>/` directories.
