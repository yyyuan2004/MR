# Experiments

## Default experiment

`configs/default.yaml` defines the default experiment:

- 200 synthetic 64x64 random-ellipse test signals; first 120 train, next 80 test.
- Measurement budget: 25% of the frequency domain (1024 of 4096 locations),
  with 2% of the budget forced onto the frequency-domain center.
- Eleven masks: point-wise `uniform_random`, `variable_density`,
  `multilevel_random`, `aopt_greedy`, `psf_penalized_aopt_greedy`; line-wise
  (Cartesian columns) `equispaced_lines`, `variable_density_lines`,
  `line_aopt`, `line_subspace_leakage`, `spectrum_energy_greedy`,
  `recon_in_loop_greedy`.
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
- **variable_density_lines** — whole Cartesian columns drawn with polynomially
  decaying density; a forced center block plus a partial column meet the
  budget exactly.
- **multilevel_random** — dyadic radial annuli with per-level budgets
  (denser toward the center), uniform random within each level.
- **aopt_greedy** — greedy Bayesian A-optimal selection under a diagonal
  Gaussian prior in the frequency domain. The prior spectrum is a radial power law fitted
  to the train split. Adding location k reduces the expected posterior MSE by
  `s_k^2 / (s_k + noise_var)`; the greedy loop always adds the largest
  remaining gain.
- **psf_penalized_aopt_greedy** (previously `artifact_aware_greedy`; the old
  name still works) — the A-optimal gain penalized by the PSF max sidelobe:
  `score = gain / max_gain - beta * max_sidelobe`. The prior enters through
  the gain term and arrangement coherence through the penalty. Candidates
  come from a hybrid pool (top gain, radius-weighted random draws, the
  boundary ring of the current support, and sidelobe-reduction candidates);
  a pure top-gain pool clusters on the low-frequency disk boundary and
  collapses this mask onto plain A-opt. Use
  `scripts/05_artifact_aware_mask_search.py --beta-sweep` to pick `beta`; the
  sweep reports the Jaccard overlap with plain A-opt per beta.
- **line_aopt** — A-optimal selection of whole columns; with a diagonal prior
  the column gain is the sum of its per-location gains, so greedy selection
  over columns is exact.
- **line_subspace_leakage** — columns chosen to minimize wavelet-subspace
  leakage: column gain = sum over wavelet subbands of (training energy in the
  subband) x (fraction of the subband's spectral mass on that column). This
  is an information-coverage criterion tied to the reconstruction basis, not
  raw Fourier energy.
- **spectrum_energy_greedy** — columns ranked by empirical mean spectral
  energy of the train split (the line-wise analogue of `data_driven_greedy`).
- **recon_in_loop_greedy** — columns scored by actually running a cheap
  wavelet-ISTA on a small training batch for every candidate and keeping the
  column with the lowest reconstruction error. This is the only criterion
  that accounts for what the nonlinear method can re-impute from the null
  space.
- **data_driven_greedy** — point-wise: uses the empirical mean spectral
  energy of the train images instead of a fitted prior. By Parseval,
  zero-filled MSE equals the unmeasured spectral energy, so each greedy step
  adds the unmeasured location with the largest measured mean `|X_k|^2`. Add
  it to `mask.types` to include it in the comparison.

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

## Subspace / manifold priors (scripts 08-10)

The prior can also be a low-dimensional model of the signal class itself,
abstracted as an N x d basis matrix: a linear subspace (`fit_subspace`, top
SVD modes of the train split) or the Jacobian of a fixed generator at a
reference latent point (`generator_jacobian_basis`) — both share one
selection and reconstruction code path.

- **Design**: `greedy_subspace_aoptimal` minimizes
  `trace((Phi_Omega^H Phi_Omega + ridge I)^-1)` for `Phi = F B`, one
  frequency-domain row at a time, with O(dN)-per-step Sherman-Morrison
  updates; the regularized trace is monotonically non-increasing (tested).
  `beta > 0` adds the min-max-normalized PSF max-sidelobe penalty.
- **Reconstruction**: `subspace_recon` solves the prior-constrained least
  squares in closed form. Its output lies in the span of the basis, *not* in
  the observed subspace — so this linear method has
  `recon_nullspace_norm > 0` by construction, unlike zero-filling and Wiener.
  The imputation is only as faithful as the prior: with a d-dimensional basis
  capturing a fraction q of signal energy, the model bias floors the error at
  roughly the un-captured (1 - q) energy. `generative_recon` replaces the
  closed form with latent-space gradient descent on a generator.
- **Learned arm**: `scripts/12_train_unet.py` trains a U-Net post-processor on
  zero-filled magnitudes (training split only) and registers it as
  `unet_post`. It imputes null-space content like the other priors, but it
  enforces no data consistency — there is no analogue of ISTA's `final_dc`
  step — so its `consistency_norm` is substantially nonzero (on the default
  run, ~4.6x the zero-filled baseline, which only carries measurement noise).
  That is a reported property of post-processing, not a defect, and it is
  precisely the kind of distinction the decomposition exists to expose.
- **Metric**: `subspace_nullspace_leakage(B, mask)` — the fraction of the
  basis energy falling on unmeasured locations, the subspace analogue of
  `aliasing_energy_ratio`. On the default run it ranks all compared masks,
  including the learned line mask from script 09, in the same order as the
  measured subspace-reconstruction error (script 10 prints the two rankings
  and their Spearman correlation; note the sample is small — a handful of
  masks).

## Measurement operators

The decomposition never uses the Fourier transform directly — it only needs
the orthogonal projector `P = A^H A` onto the observed subspace. `operators.py`
makes the operator pluggable and ships two:

- `FourierOperator` (default, `A = M F`): the mask indexes frequencies.
- `InpaintingOperator` (`A = M`): the mask indexes signal samples, so `P` is
  already diagonal in the signal domain and the null space is literally the
  unmeasured pixels.

`decompose_error(..., operator=...)` accepts either; every identity
(complementarity, the Pythagorean split, Hermitian idempotent `P`) is tested
for both. The point is that "whether a prior helps is governed by null-space
structure" is a statement about the operator's null space, not about Fourier
sampling.

## Budget sweep and the phase diagram

At 4x acceleration the entire low-frequency block fits inside the budget, so
essentially every sensible design takes the same energy-dense core and
sampling design barely matters. The capture-energy / control-coherence
trade-off only becomes real once the budget cannot hold that core, which is
what 8x and 12x probe. `scripts/11_budget_sweep.py` sweeps the acceleration
factor over the full parameterized mask family and concatenates the per-budget
argumentation tables into `metrics/budget_sweep.csv`, printing three decisive
numbers per budget: whether the PSF penalty has separated from plain
A-optimal (Jaccard), whether the spectral energy score still predicts the
nonlinear error, and whether any null-space norm predicts the prior's payoff.

`scripts/13_phase_diagram.py` renders that table as the summary figure:
acceleration on x, mask coherence on y, and the prior's PSNR gain as a
diverging color scale whose neutral midpoint is pinned exactly at zero, with
the zero level drawn as an explicit contour. The contour is the boundary
between "the prior helps" and "the prior hurts".

## Statistical power

Rank correlations over a handful of masks are not evidence. The default
comparison in script 07 uses 11 masks, which is enough to see a strong effect
(`mask_score` vs zero-filled MSE) but far too few to call a weak one
significant — an observed rho near 0.4 at n = 11 has p ≈ 0.2 and supports no
conclusion. `experiment.build_mask_family` therefore sweeps each generator's
free parameters (density decay, level structure, penalty weight, random seed)
to produce ~31 masks by default, and every correlation row records `n_masks`
alongside rho and p. Widen `mask.family.*` to push n further.

## Argumentation table

`metrics/summary.csv` is the results table; `metrics/argumentation.csv` is the
argumentation table. Each row places a mask's design-time scores — computable
before any measurement is simulated — next to its measured outcomes:

- Predictors: `mask_score` (expected zero-filled MSE under the train
  spectrum), `wavelet_leakage` (energy-weighted null-space leakage of the
  reconstruction basis), `weighted_max_sidelobe` (max sidelobe of the
  prior-weighted PSF), `psf_max_sidelobe` (plain coherence).
- Outcomes: `mse_zero_filled`, `mse_wavelet_ista`, `psnr_gain_ista` (the
  nonlinear method's improvement over zero-filling), `ista_nullspace_norm`.

`metrics/argumentation_correlations.csv` reports Spearman rank correlations of
every predictor against every outcome across masks. Note that plain PSF
coherence alone cannot rank masks — it ignores where signal energy sits —
which is why the spectrum-weighted and leakage predictors exist. The
`psf_sidelobe_energy` column of the PSF metrics is budget-dominated (by
Parseval it is fixed at `1 - budget/N` regardless of arrangement) and is kept
only for completeness; do not use it for ranking.

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
