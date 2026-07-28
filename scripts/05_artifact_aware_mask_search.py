#!/usr/bin/env python3
"""Build the PSF-penalized A-optimal greedy mask and evaluate it on the test
split. (This mask type was previously named artifact_aware_greedy; the old
name is still accepted in configs.)

With --beta-sweep, additionally sweep the sidelobe penalty weight over
greedy.beta_sweep and report, per beta: Jaccard overlap with plain A-opt,
weighted and plain PSF max sidelobes, mask score, and quick reconstruction
metrics on a test subset (metrics/beta_sweep.csv).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mrsim import artifacts, experiment, greedy, masks, recon
from mrsim.config import load_config, run_dir, save_config_snapshot, seed_everything


def beta_sweep(
    cfg: dict, train: torch.Tensor, validation: torch.Tensor, run: Path
) -> pd.DataFrame:
    """Tune beta on validation data; the test split is never touched here."""
    g = cfg.get("greedy", {})
    betas = [float(b) for b in g.get("beta_sweep", [0.0, 0.5, 1.0, 2.0, 4.0])]
    shape, n_samples, n_center = experiment.mask_budgets(cfg)
    noise_var = experiment.greedy_noise_var(cfg)
    n_candidates = int(g.get("n_candidates", 32))
    seed = int(cfg["seed"])

    prior = experiment.fitted_power_law_spectrum(train)
    prior_mean, prior_variance, train_power = experiment.frequency_prior_statistics(train)
    aopt = greedy.greedy_a_optimal(prior, n_samples, noise_var=noise_var, n_center=n_center)
    if validation.shape[0] == 0:
        raise ValueError("beta sweep requires a non-empty validation split")
    subset = validation[: min(16, validation.shape[0])]
    noise_std = float(cfg.get("measurement", {}).get("noise_std", 0.0))
    common_noise = None
    if noise_std > 0.0:
        common_noise = recon.sample_complex_noise_like(
            subset,
            noise_std,
            generator=torch.Generator().manual_seed(seed + 501),
        )

    rows = []
    for beta in betas:
        mask = greedy.greedy_psf_penalized_aopt(
            prior, n_samples,
            noise_var=noise_var, beta=beta, n_candidates=n_candidates, n_center=n_center,
            rng=np.random.default_rng(seed), show_progress=True,
        )
        reconstructed = experiment.reconstruct_all(
            subset,
            mask,
            cfg,
            noise=common_noise,
            spectrum=prior_variance,
            prior_mean=prior_mean,
            return_measurements=True,
        )
        recons, measurements = reconstructed
        quick = pd.DataFrame(
            experiment.metrics_rows(
                recons,
                subset,
                mask,
                f"beta={beta:g}",
                measurements=measurements,
            )
        )
        psnr = quick.groupby("method")["psnr"].mean()
        rows.append(
            {
                "beta": beta,
                "jaccard_vs_aopt": masks.jaccard(mask, aopt),
                **artifacts.spectrum_weighted_psf_metrics(mask, train_power),
                **artifacts.psf_metrics(mask),
                "mask_score": artifacts.expected_zero_filled_mse(
                    mask, train_power, noise_std=noise_std
                ),
                "psnr_zero_filled": float(psnr.get("zero_filled", float("nan"))),
                "psnr_wavelet_ista": float(psnr.get("wavelet_ista", float("nan"))),
            }
        )
    sweep = pd.DataFrame(rows)
    out = run / "metrics" / "beta_sweep.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    sweep.to_csv(out, index=False)
    return sweep


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--beta-sweep", action="store_true", help="run the penalty-weight sweep")
    args = parser.parse_args()

    cfg = load_config(args.config)
    rng = seed_everything(int(cfg["seed"]))
    run = run_dir(cfg)
    save_config_snapshot(cfg, run, "05_artifact_aware_mask_search", cli_args=vars(args))

    images = experiment.load_or_generate_dataset(cfg, run)
    train, validation, test = experiment.train_validation_test_split(images, cfg)

    mask_dict = experiment.build_masks(["psf_penalized_aopt_greedy"], cfg, train, rng)
    prior_mean, prior_variance, _ = experiment.frequency_prior_statistics(train)
    frame = experiment.evaluate_masks(
        mask_dict,
        test,
        cfg,
        run,
        prefix="psf_penalized",
        spectrum=prior_variance,
        prior_mean=prior_mean,
    )

    summary = frame.groupby(["mask", "method"])[["psnr", "ssim", "nrmse"]].mean()
    print(summary.round(4).to_string())
    print(f"\nmetrics: {run / 'metrics' / 'psf_penalized_metrics.csv'}")

    if args.beta_sweep:
        sweep = beta_sweep(cfg, train, validation, run)
        print("\nbeta sweep (beta_sweep.csv):")
        print(sweep.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
