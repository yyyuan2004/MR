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

`configs/default.yaml` generates 400 images and uses 200/100/100
train/validation/test splits. The previous 36/12/12 was buying nothing:
synthetic phantoms cost microseconds, and at 12 test images the bootstrap
intervals on PSNR were wider than the gaps between masks while the 90%
conformal coverage interval spanned [0.730, 1.000]. Summary tables now carry
percentile bootstrap intervals over images. They are *not* computed over masks:
masks in a design sweep are constructed, dependent objects, so an interval
across them would not mean anything.

Script 07 compares:

- point-wise baselines: `uniform_random`, `variable_density`,
  `multilevel_random`, and `multilevel_local_coherence`;
- full-column Cartesian baselines: `equispaced_lines` and
  `variable_density_lines`.

`multilevel_local_coherence` derives its per-level density from the measured
Fourier-wavelet local coherence rather than a hand-chosen decay exponent. It
does not win: 27.15 PSNR (95% CI 26.83-27.50) against `variable_density`'s
28.10 (27.79-28.44), with non-overlapping intervals. It acquires *more*
information by the conjugate-orbit measure (817 effective samples against 747),
which sharpens the point — uniform recovery guarantees over a sparse class and
average-case error on one distribution are different objectives, and the
coherence density spends budget at high frequencies where this signal class has
little energy. The tuned exponent is not vindicated either; it was fit to this
distribution and never compared on held-out data.

Note also that `variable_density` and `multilevel_random` have overlapping
intervals (27.79-28.44 against 27.67-28.32). The ordering between those two
should not be reported as a result.

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

The current repository evaluates both terms against known synthetic truth.

The certification question that used to sit open here has three parts with
different answers, and `mrsim/certify.py` separates them:

1. **An assumption-free null-space bound does not exist.** The feasible set
   `{x : ||A x - y|| <= eps}` has infinite diameter along `ker(A)`, so the
   supremum of the null-space error over it is unbounded at any confidence
   level. This needs no theorem.
2. **Distribution-free *conditional* coverage is unattainable** for non-atomic
   distributions with finite samples (Vovk; Barber, Candès, Ramdas and
   Tibshirani). This is a citable result, not something the repository needs to
   prove, and nothing here claims conditional validity.
3. **Marginal coverage under exchangeability is achievable**, and is
   implemented. Split-conformal calibration on the validation split yields a
   threshold covering a fresh exchangeable draw with probability at least
   `1 - alpha`. The guarantee is over the joint draw of calibration and test
   points, says nothing about any individual image, and does not survive the
   prior shift contemplated elsewhere in this document — a test confirms it
   genuinely fails under shift rather than degrading gracefully.

Beyond the projector split, `artifacts.bias_variance_decomposition` splits
expected error into bias and variance *within* each subspace. The four pieces
sum exactly to the total, and they distinguish a prior that is too strong from
noise amplification — which the observed/null split alone cannot, since a
regularized estimator and an ill-conditioned one look alike under it.

## Real signals and conjugate redundancy

The generator produces real images, so `X(-k) = conj(X(k))`. Sampling both
members of a conjugate pair acquires one complex unknown, not two. The
effective information budget is therefore the number of conjugate *orbits* a
mask touches, reported as `effective_samples` / `hermitian_redundancy` beside
the nominal count.

The two budgets come apart badly at equal nominal size. Measured on the shipped
configuration:

| mask | nominal | effective | redundancy |
| --- | --- | --- | --- |
| `uniform_random` | 1024 | 875 | 0.146 |
| `multilevel_local_coherence` | 1024 | 817 | 0.202 |
| `multilevel_random` | 1024 | 751 | 0.267 |
| `variable_density` | 1024 | 747 | 0.271 |
| `variable_density_lines` | 1024 | 737 | 0.280 |
| `equispaced_lines` | 1024 | 514 | 0.498 |

`equispaced_lines` acquires columns `{0, 4, ..., 60}`, a set closed under
`c -> (64 - c) mod 64`, so exactly half its budget lands on values already
determined. A comparison "at equal budget" spans a factor of 1.7 in effective
degrees of freedom, and the ordering runs against intuition: radially symmetric
densities cluster conjugate pairs and are penalised.

This also breaks the top-k design rule below. See `mrsim/design.py` and
`greedy.greedy_a_optimal_hermitian`.

## Exact design quantities replace the proxies

Under a Gaussian prior and Gaussian noise the Bayes MMSE and the mutual
information are available in closed form, so `rho`, `wavelet_leakage` and
`mask_score` are not needed to rank designs — they are proxies for quantities
`mrsim/design.py` computes exactly. Two consequences are now tested rather than
asserted:

- **The design problem is degenerate for a diagonal prior.** Both the A-optimal
  and D-optimal binary optima are exactly top-k rankings of the prior spectrum,
  verified against exhaustive enumeration. Whatever the default arm's greedy
  selectors are doing, they are not solving a hard problem.
- **That degeneracy is an artifact of the independent-coefficient model.** For a
  real signal, top-k buys both members of each conjugate pair. On a 4x4 grid it
  is more than 50% worse than the exhaustive optimum; at the shipped scale it
  reaches 513 effective samples against 1024, and `greedy_a_optimal_hermitian`
  cuts the exact Bayes MMSE by 46.6%.

`optimality_gap` reports a mask's distance from the exact binary optimum in
nats. The convex (Joshi-Boyd) relaxation is computed too, but its distance from
the binary optimum is an **integrality gap** — roughly 30 nats on an 8x8
example where top-k is provably optimal — and must not be read as a mask being
suboptimal. Relatedly, the `(1 - 1/e)` greedy guarantee applies to the monotone
submodular logdet objective, **not** to the A-optimal trace objective the
shipped greedy selectors actually optimize.

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

Both conclusions drawn from this table have since been qualified. Read them
with the two corrections below.

**The coverage metrics are interchangeable *on this family*, and the reason is
not the one it looks like.** Both `rho` and `wavelet_leakage` are exactly affine
in the mask, `c - <M, w>` — each wavelet subband's spectral mass sums to one,
so the identity holds to machine precision. But affineness alone does not force
rank collapse: the two weight vectors have cosine similarity only 0.30, and over
150 unstructured masks at the same budget the rank correlation is -0.36. The
collapse is a property of the *family*, whose members are all parameterized by
how tightly they concentrate toward low frequency; both functionals track that
single latent axis at |rho| ~ 0.97. Raw mask space is not low-dimensional here
(PC1 explains 14% of the variance). A family varying something other than
radial concentration could separate them, so any redundancy claim has to name
the family it holds over. `tests/test_linear_functional_collapse.py` keeps the
negative control.

**sigma_min is not redundant with coverage; the single fixed support made it
look that way.** Evaluating conditioning on one aggregated support turns a
union-of-subspaces problem back into a single-subspace one, which is the
linear-Gaussian regime where conditioning largely follows coverage. Drawing a
distribution of supports instead:

| support | rho vs sigma_min | reading |
| --- | --- | --- |
| single aggregated (above) | +0.963 | redundant with coverage |
| 16 drawn iid by energy | +0.477 | carries independent information |
| 16 drawn tree-structured | +0.387 | carries independent information |

The tree-structured model exists because wavelet coefficients are not
independent across scales — large ones persist along parent-child chains — and
it moves the numbers materially, roughly doubling median sigma_min for the
poorly conditioned point masks. So clustered supports are better conditioned
under those masks than scattered ones.

The rank-deficiency observation survives both corrections: equispaced lines
leave 146 of 256 active-support directions exactly unrecoverable,
variable-density lines 13-50, and every point-wise mask none. An earlier run
without the support cap reported coherence and sigma_min as independent
(rho = +0.09); that number was an artifact of rank-ordering numerical noise
among deficient masks.

### Per-subband sigma_min profile

The scalar sigma_min collapses direction and scale; the per-subband profile
(`sigma_min_<band>` columns in `metrics/axis_precheck.csv`,
`plots/subband_sigma_min.png`) restores both, because 2-D wavelet orientation
bands encode direction and the level hierarchy encodes scale. On the default
family it turns each headline number into an explanation: equispaced lines
are rank-deficient in the approximation band and every coarse detail band
(hence 146 dead directions), while the line-selection masks show a sharp
orientation asymmetry — e.g. level-2 vertical sigma_min of 0.002 against
level-2 horizontal 0.77 for `line_subspace_leakage` — which is exactly the
statement "column sampling conditions one orientation and collapses the
other" as a measured quantity. Point masks degrade smoothly from coarse to
fine scales instead. Level 1 denotes the coarsest detail level.

### Dispersion of sigma_min at fixed coverage

Rank statistics alone cannot say whether conditioning is informative *given*
coverage, so script 13 also bins the full-rank masks into fixed-rho quantile
bins and reports the within-bin spread of sigma_min
(`metrics/axis_precheck_dispersion.csv`, plus a family-colored
`plots/rho_vs_sigma_min.png`). On the default family the answer is
regime-dependent: at low-to-mid coverage the within-bin spread is 1.8-2.3
decades of sigma_min at essentially fixed rho (conditioning carries real
information beyond coverage there), while in the highest-coverage bins the
spread collapses to 0.2-0.3 decades (coverage determines conditioning for
energy-dense masks). Overall, 29% of the log10 sigma_min variance lies within
bins. The global rank correlation is also support-size dependent (+0.77 at
|S| = 384 vs +0.97 at |S| = 256), so any redundancy claim must state |S|.

## Evaluated and not adopted

Recorded with reasons so they are not re-proposed. Each was checked against
this codebase rather than dismissed on principle.

- **Adaptive / sequential sampling.** For a linear-Gaussian model the posterior
  covariance `(Sigma^-1 + sum_k w_k phi_k phi_k^H / sigma^2)^-1` does not depend
  on the observed values, so the optimal sequential design equals the optimal
  batch design and adaptivity gains exactly nothing. It becomes meaningful only
  under non-Gaussian priors with unknown support, and even there adaptivity does
  not improve minimax rates for sparse estimation beyond constant factors.
- **Minimising global coherence.** Measured here, the Fourier-db4 global
  coherence is `mu = 8` against an incoherent floor of 1, and the classical
  bound `m >~ mu^2 k log N` then demands roughly 136000 measurements against an
  ambient dimension of 4096 — vacuous. The usable structure is that coherence
  varies by frequency (approx 8.0 down to level-3 2.0), which is what motivates
  `multilevel_local_coherence` above.
- **Merging the point and line strata.** Rank deficiency is the discriminating
  non-linear structure and it comes almost entirely from the line masks;
  averaging over the strata would destroy it. The two are also different
  feasible sets, so comparing them at equal cardinality is a category error
  unless the cost of the constraint is itself the object of study.
- **Shannon rate-distortion as the performance floor.** `R(D)` constrains bits;
  the constraint here is the number of linear measurements, chosen by the
  designer rather than by an encoder that has seen the signal. The correct floor
  for the Gaussian case is the Bayes MMSE of the best m-row design, which
  `mrsim/design.py` computes exactly. The sparse-class rate `sigma^2 k log(N/k)/m`
  is an order bound with unspecified constants and can only be checked for
  scaling, not plotted against measured error.
- **"The null space is determined by the prior's support."** `ker(A)` depends on
  `A` alone; changing the prior changes only whether the null-space component is
  *inferable*. The existing wording in this document is correct as it stands.
- **End-to-end joint optimization of sampling and reconstruction.** It deepens
  the attribution confound this revision is removing — error can no longer be
  assigned to design versus algorithm — so it belongs after, not before, the
  measurement fixes. The LOUPE arm remains quarantined.

## Future work

- Treat the **operator/prior pair** as the experimental axis rather than the
  mask: add Gaussian, subsampled Hadamard, and rotated-Fourier operators to
  `mrsim/operators.py` and test the claim that design gains scale with how
  non-diagonal the prior covariance is in the measurement basis. The exact
  quantities in `mrsim/design.py` make this directly measurable.
- A **phase diagram** over undersampling ratio and effective sparsity, swept
  across several `N`, is the honest route to the regime-transition claim this
  document currently disclaims.
- Performance: the wavelet soft-threshold round-trips through NumPy per image
  per iteration, and `greedy_psf_penalized_aopt` recomputes a full FFT per
  candidate where the rank-one plane-wave update already derived in
  `_sidelobe_reduction_candidates` would serve.
