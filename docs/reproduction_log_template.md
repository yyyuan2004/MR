# Reproduction log

Copy this file for each run and complete every field.

## Environment

- Date:
- Operator:
- Git commit: `git rev-parse HEAD`
- Config fingerprint / run directory:
- Python version: `python --version`
- Package versions: `pip freeze | grep -E "torch|numpy|scipy|scikit-image|PyWavelets|matplotlib|pandas|PyYAML"`
- OS / CPU / RAM:
- GPU used (expected `no` for default and smoke):

## Protocol

- Config file:
- Purpose: smoke / baseline evidence / experimental ablation
- Signal generator or dataset:
- Train / validation / test sizes:
- Same-distribution or named shift:
- Forward operator:
- Single-coil or multi-coil:
- Noise model and level:
- Mask geometry: point / full Cartesian line
- Sampling fraction and exact sample count:
- Ground truth used for model selection (must be `no` for test truth):
- Command(s):

```bash

```

- Seed(s):
- Wall-clock time:

## Outputs

- Run directory:
- Config manifest present:
- `metrics/summary.csv` present:
- Key results:

| acquisition stratum | mask | method | psnr_mean | ssim_mean | nrmse_mean |
| --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |

Do not combine point and line masks into one headline rank without explicitly
justifying the acquisition constraint.

## Information audit

- Operator/measurement-observable quantities reported:
- Prior-derived design quantities reported:
- Oracle test-truth quantities reported:
- Any oracle quantity described as a runtime certificate (must be `no`):
- If `rho_pi(A)` is reported, was it estimated as a ratio of pooled expected
  energies rather than one minus an unweighted mean of per-image ratios:

## Comparison against reference

- Reference run / commit:
- Metrics match within tolerance (state tolerance):
- Deviations and suspected causes:
- Repeated seeds and uncertainty summary:
- Any inferential claim:
- If yes, pre-specified test, multiplicity correction, and confirmatory set:

## Notes

-
