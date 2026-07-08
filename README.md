# MR

Measurement-mask design and comparison toolkit for undersampled
frequency-domain inverse problems. Generates synthetic test signals, builds
frequency-domain measurement masks (random, structured, and
greedy/data-driven), reconstructs with linear and iterative methods, and
compares masks with reconstruction and artifact metrics, including an
observed-subspace / null-space error decomposition.

## Installation

Requires Python 3.12+ (runs on 3.11 as well).

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

CPU-only PyTorch is sufficient:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Quickstart

Run the default experiment (200 synthetic 64x64 images, 5 mask types,
zero-filled, Wiener, and wavelet-ISTA reconstruction) from the repository root:

```bash
python scripts/07_compare_all_masks.py --config configs/default.yaml
```

Outputs land in `runs/default/`. Run the tests with:

```bash
pytest
```

## Repository structure

```
configs/            YAML experiment configs
mrsim/              library code
  config.py         config loading, run dirs, seeding
  data.py           synthetic test signal generation
  fft_ops.py        centered orthonormal FFT, forward/adjoint, projector
  masks.py          baseline mask generators, point- and line-wise (exact budgets)
  greedy.py         greedy selection: A-optimal, PSF-penalized, data-driven,
                    line-wise, wavelet-leakage, reconstruction-in-the-loop
  progress.py       dependency-free progress reporting
  recon.py          measurement simulation; zero-filled, Wiener, wavelet-ISTA reconstruction
  metrics.py        PSNR / SSIM / NRMSE / MSE
  artifacts.py      decomposition, artifact maps, PSF metrics, mask scores
  viz.py            image grids, PSF plots, scatter plots
  unet.py           optional compact U-Net (not used by the default pipeline)
  experiment.py     shared plumbing for the numbered scripts
scripts/            numbered experiment scripts (01-07), run from repo root
tests/              pytest suite
docs/               usage, experiment descriptions, reproduction log template
runs/               experiment outputs (gitignored)
```

## Running synthetic experiments

Scripts can be run individually; each loads a YAML config, writes outputs under
`runs/<experiment_name>/`, and saves a JSON snapshot of the config it used:

```bash
python scripts/01_generate_synthetic_data.py --config configs/default.yaml
python scripts/02_make_baseline_masks.py     --config configs/default.yaml
python scripts/03_reconstruct_and_evaluate.py --config configs/default.yaml
python scripts/04_aopt_greedy_mask.py        --config configs/default.yaml
python scripts/05_artifact_aware_mask_search.py --config configs/default.yaml
python scripts/06_greedy_data_driven_mask.py --config configs/default.yaml
python scripts/07_compare_all_masks.py       --config configs/default.yaml
```

See `docs/usage.md` for the config reference and `docs/experiments.md` for what
each mask and score means.

## Comparing sampling masks

`scripts/07_compare_all_masks.py` builds every mask listed under `mask.types`
in the config, reconstructs the test split with zero-filled, Wiener, and
wavelet-ISTA methods,
and writes:

- `runs/<exp>/metrics/compare_metrics.csv` — per-image metrics per mask/method
- `runs/<exp>/metrics/summary.csv` — aggregated table with PSF metrics and mask scores
- `runs/<exp>/plots/score_vs_error.png` — predicted mask score vs measured error

## Inspecting artifact maps

Artifact maps are the pointwise magnitude of the complex reconstruction error
`|recon - truth|`. The comparison script saves grids of five representative
test images (spread across the difficulty range) per mask and method under:

- `runs/<exp>/artifact_maps/<mask>_<method>.png`
- matching reconstructions in `runs/<exp>/recon/<mask>_<method>.png`
- PSF plots per mask in `runs/<exp>/psf/<mask>_psf.png`
