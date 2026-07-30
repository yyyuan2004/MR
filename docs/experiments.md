# Experiment interpretation

## Current empirical scope

The checked-in default experiment is an in-distribution synthetic benchmark:

- signals are random ellipse images;
- train, validation, and test splits use the same generator;
- measurements use an ideal single-coil Cartesian Fourier operator;
- noise is simulated circular complex Gaussian k-space noise;
- all images are 64x64 and all default methods run on CPU.

This setup is useful for checking algebra, data flow, and controlled behavior.
It is not evidence for clinical MRI, multi-coil reconstruction, CT,
inpainting, cross-anatomy transfer, cross-device transfer, or robustness to a
misspecified prior. The operator abstraction in `mrsim/operators.py` is an
extension point and is exercised by unit tests; it does not enlarge the
empirical scope of the default study.

## Default comparison

`configs/default.yaml` generates 60 images and uses 36/12/12
train/validation/test splits. Script 07 compares:

- point-wise baselines: `uniform_random`, `variable_density`, and
  `multilevel_random`;
- full-column Cartesian baselines: `equispaced_lines` and
  `variable_density_lines`.

All masks use a 25% point budget. Because 1024 samples at 64x64 equal 16 full
columns, the two shipped line masks contain only complete columns. Point and
line results should be reported separately: equal point counts do not make
their acquisition constraints equivalent.

The reconstruction methods are zero filling, a diagonal-Gaussian posterior
mean using train-set frequency mean and variance, and wavelet ISTA. With
nonzero measurement noise, the shipped config sets
`wavelet_ista.final_dc: false`; a hard final replacement would reinsert the
noisy measured coefficients exactly.

## Operator/statistics decomposition

For the ideal masked Fourier operator, let

```text
A = M F,        P = A* A = F* M F,
e = R(y) - x.
```

Because `P` is an orthogonal projector,

```text
e = P e + (I-P)e,
||e||^2 = ||P e||^2 + ||(I-P)e||^2.
```

The first term lies in the observed subspace. Given a noise bound and the
known operator, it can be related to the observable residual
`y - A R(y)`. The second term lies in the null space: two signals on the same
measurement fiber have identical noiseless measurements, so resolving this
term requires assumptions or prior information.

The current repository evaluates both terms against known synthetic truth. It
does not yet turn the residual relation into a calibrated per-sample
certificate, and it does not prove a distribution-free conditional-coverage
impossibility theorem. The decomposition is therefore an exact diagnostic
identity, while the broader certification claims remain theory work outside
the present experiment.

## The certifiable-fraction notation

A distribution-dependent population quantity can be defined as

```text
rho_pi(A) = E_pi ||P x||^2 / E_pi ||x||^2.
```

It measures how much signal energy the operator exposes under a specified
population. Its complement is the population energy placed in the operator's
null space.

The per-image CSV field

```text
oracle_unsampled_energy_ratio(x, A) = ||(I-P)x||^2 / ||x||^2
```

is an oracle evaluation quantity because it needs `x`. For one fixed image it
equals `1 - ||Px||^2/||x||^2`. Across a dataset, however,
`1 - mean(oracle_unsampled_energy_ratio)` is generally not identical to
`rho_pi(A)`, which is a ratio of expectations, unless image energies are
constant or the estimator uses pooled numerator and denominator sums.

The argumentation table reports the pooled train-set estimate as
`prior_observable_energy_fraction`. Estimating `rho_pi(A)` from the train
split makes it a prior-derived design diagnostic. Estimating it from test
truth makes it an oracle evaluation metric. Neither usage is a
distribution-free property of `A` alone.

## Observable, prior-derived, and oracle quantities

| Class | Examples | Valid interpretation |
| --- | --- | --- |
| Operator/measurement-observable | mask, budget, PSF, measured data, `||y-A x_hat||` | Available without test truth; residual bounds still require a noise statement. |
| Prior-derived design-time | train second moment, `prior_observable_energy_fraction`, `mask_score`, weighted PSF, wavelet/subspace leakage | Computable before test evaluation, but inherits the assumed training prior. |
| Oracle evaluation | `complex_mse`, `magnitude_mse`, PSNR/SSIM/NRMSE, `oracle_unsampled_energy_ratio`, `oracle_nullspace_error_norm`, `oracle_observed_subspace_error_norm`, `oracle_truth_nullspace_norm` | Available only because synthetic test truth is known. |

The observable `measurement_residual_norm` is `||y-A x_hat||`. The separate
field `oracle_observed_subspace_error_norm` is `||P(x_hat-x)||` and must remain
in the oracle category.

## Stable baseline masks

- `uniform_random`: samples individual coefficients uniformly without
  replacement.
- `variable_density`: samples individual coefficients with probability
  decaying away from the Fourier center.
- `multilevel_random`: assigns point budgets to radial annuli and samples
  within each annulus.
- `equispaced_lines`: samples Cartesian columns on a regular grid.
- `variable_density_lines`: samples Cartesian columns with a center-biased
  distribution.

Line-mask helpers never partially fill a column. A non-divisible point budget
is floored to the largest feasible number of complete columns, and output
tables record the actual sample count and acceleration.

## Experimental opt-in methods

The following components are research prototypes rather than default
baselines:

- diagonal-prior A-optimal and PSF-penalized greedy masks;
- reconstruction-in-the-loop selection;
- centered affine Gaussian subspaces (PCA covariance factors) and
  generator-Jacobian bases;
- LOUPE-style learned masks;
- U-Net post-processing;
- broad parameter and acceleration sweeps.

Their settings remain in `configs/default.yaml` so the associated scripts can
be invoked explicitly. They are excluded from `mask.types`, are not run by CI,
and should not be cited as validated results without:

1. independent random seeds and uncertainty reporting;
2. validation-only model and hyperparameter selection;
3. a fixed comparison within the same acquisition geometry;
4. an explicit train/test shift protocol when robustness is the claim;
5. an untouched confirmatory test set.

## What the current default can and cannot support

The default run can support statements such as:

- the projector decomposition is numerically consistent in the synthetic
  masked-Fourier setting;
- mask geometry and oracle reconstruction metrics can be generated
  reproducibly;
- point and full-line baselines behave differently under their respective
  constraints.

It cannot by itself support statements such as:

- a test-time null-space error certificate is observable;
- one mask is robust to prior misspecification;
- accuracy-optimal and certifiability-optimal designs provably separate;
- a regime transition has been established;
- a nominal rank-correlation p-value supports a confirmatory claim.

Those claims require additional theory and experiments rather than stronger
wording around the current in-distribution run.

## Recommended evidence sequence

Use the repository in the following order:

1. Run `configs/smoke.yaml` only as an end-to-end wiring check.
2. Run `configs/default.yaml` and verify the deterministic projector identity,
   data splits, full-line budgets, and output manifests.
3. Report point-mask and line-mask baselines separately.
4. Add repeated seeds before comparing stochastic masks.
5. Define a train/test distribution shift before evaluating prior robustness.
6. Reserve learned, subspace, and LOUPE-style arms for explicitly labeled
   ablations until they pass the same protocol.

Correlation CSVs produced by the scripts are descriptive diagnostics. The
repository has no confirmatory statistical analysis pipeline.

## Coverage and recoverability diagnostics (scripts 11 and 13)

The design chain is Mask -> Representation -> Recoverability -> Error, and the
middle link is a property of the measurement-operator x representation pair,
not of any reconstruction algorithm. Energy metrics (rho, wavelet leakage)
measure *coverage* of the representation; they cannot measure *conditioning*.
`scripts/13_axis_precheck.py` therefore computes, alongside the coverage
metrics, the full singular spectrum of the restricted operator A F W*_S on the
empirically active wavelet support, and three plots make each link visible:

- `plots/radial_coverage.png` — radial sampling density per mask with the
  variable-density kernel at decay 2 as a reference, plus the radial profile
  of rho (per-radius observed energy fraction). Also emitted per budget by
  `scripts/11_budget_sweep.py`.
- `plots/subband_leakage.png` — per-subband leakage heatmap; the
  energy-weighted row total reproduces `wavelet_leakage_score` exactly
  (tested). Also emitted per budget by script 11.
- `plots/singular_spectra.png` — the *entire* singular spectrum per mask, not
  just sigma_min. The support size is capped at half the measurement budget:
  sigma_min of an m x |S| matrix is structurally zero once |S| > m, and near
  that boundary it measures nothing. Rank-deficient masks are excluded from
  sigma_min correlations (ranking values at the numerical floor is noise).

Measured on the default 31-mask family (24 full-rank, 7 rank-deficient):

| pair | Spearman rho |
| --- | --- |
| rho vs PSF max sidelobe | +0.433 |
| rho vs wavelet_leakage | -0.994 |
| rho vs sigma_min (full-rank masks) | +0.967 |
| PSF max sidelobe vs sigma_min (full-rank masks) | +0.973 |

Two conclusions. First, the coverage metrics are interchangeable: rho and
wavelet leakage are near-perfect mirrors, so a 2-D diagnostic should never
spend both axes on them. Second, among full-rank masks sigma_min adds nothing
beyond coverage either — the discriminating recoverability signal on this
family is the *rank deficiency itself*: equispaced lines leave 146 of 256
active-support directions exactly unrecoverable, variable-density lines 13-50,
and every point-wise mask none. That count (and the shape of the spectrum's
tail) is the non-energy axis; sigma_min ranked among full-rank masks is not.
An earlier run without the support cap reported coherence and sigma_min as
independent (rho = +0.09); that number was an artifact of rank-ordering
numerical noise among deficient masks and is superseded by the table above.
