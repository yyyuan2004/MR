#!/usr/bin/env python3
"""Default experiment: build every configured mask, reconstruct the test split
with zero-filled, Wiener, and wavelet-ISTA methods, and compare masks.

Outputs (under runs/<experiment_name>/<config-hash>/):
  metrics/compare_metrics.csv        per-image metrics for every mask x method
  metrics/compare_psf_metrics.csv    PSF metrics per mask
  metrics/summary.csv                aggregated results table
  metrics/argumentation.csv          design-time scores vs measured outcomes
  metrics/argumentation_correlations.csv  Spearman rank correlations
  plots/score_vs_error_<family>.png  stratified score vs measured error
  plots/psf_profiles.png             center-row PSF profile overlay
  plots/zoom_comparison.png          crop-and-zoom comparison (wavelet ISTA)
  masks/, psf/, recon/, artifact_maps/   per-mask images
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mrsim import artifacts, experiment, recon, viz
from mrsim.config import load_config, run_dir, save_config_snapshot, seed_everything

PREDICTORS = [
    "mask_score",
    "prior_unobservable_energy_fraction",
    "wavelet_leakage",
    "weighted_max_sidelobe",
    "psf_max_sidelobe",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--tune-ista", action="store_true",
        help="select the ISTA threshold per mask on the validation split",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    rng = seed_everything(int(cfg["seed"]))
    run = run_dir(cfg)
    save_config_snapshot(cfg, run, "07_compare_all_masks", cli_args=vars(args))

    images = experiment.load_or_generate_dataset(cfg, run)
    train, validation, test = experiment.train_validation_test_split(images, cfg)
    mask_names = list(cfg["mask"]["types"])
    print(f"building masks: {', '.join(mask_names)}")
    mask_dict = experiment.build_masks(mask_names, cfg, train, rng)

    # Centered prior statistics drive Wiener; the second moment predicts ZF MSE.
    prior_mean, prior_variance, train_power = experiment.frequency_prior_statistics(train)
    include_unet = bool(
        cfg.get("experimental", {}).get("include_unet_in_comparisons", False)
    )
    unet = experiment.load_unet(run, cfg=cfg) if include_unet else None
    if unet is not None:
        print("including experimental U-Net post-processor as 'unet_post'")
    if args.tune_ista and validation.shape[0] == 0:
        raise ValueError("--tune-ista requires data.n_val > 0")
    frame = experiment.evaluate_masks(
        mask_dict, test, cfg, run, prefix="compare",
        spectrum=prior_variance, prior_mean=prior_mean, unet_model=unet,
        val_images=validation if args.tune_ista else None,
    )

    scores = {
        name: artifacts.expected_zero_filled_mse(
            mask,
            train_power,
            noise_std=float(cfg.get("measurement", {}).get("noise_std", 0.0)),
        )
        for name, mask in mask_dict.items()
    }
    summary = (
        frame.groupby(["mask", "method"])[
            [
                "magnitude_mse",
                "complex_mse",
                "psnr",
                "ssim",
                "nrmse",
                "oracle_unsampled_energy_ratio",
                "measurement_residual_norm",
            ]
        ]
        .agg(["mean", "std"])
    )
    summary.columns = ["_".join(col) for col in summary.columns]
    summary = summary.reset_index()
    summary["mask_score"] = summary["mask"].map(scores)
    mask_metadata = pd.DataFrame(
        [
            {
                "mask": name,
                "actual_n_samples": int(mask.sum()),
                "actual_acceleration": float(mask.size / mask.sum()),
                "acquisition_family": experiment.acquisition_family(mask),
            }
            for name, mask in mask_dict.items()
        ]
    )
    summary = summary.merge(mask_metadata, on="mask", how="left")
    psf_frame = pd.read_csv(run / "metrics" / "compare_psf_metrics.csv")
    summary = summary.merge(psf_frame, on="mask", how="left")
    summary.to_csv(run / "metrics" / "summary.csv", index=False)

    # Argumentation table: do the design-time scores predict the outcomes?
    ista_cfg = cfg.get("recon", {}).get("wavelet_ista", {})
    wavelet = str(ista_cfg.get("wavelet", "db4"))
    levels = int(ista_cfg.get("levels", 3))
    mass = artifacts.subband_spectral_mass(train_power.shape, wavelet=wavelet, levels=levels)
    energies = artifacts.subband_energies(
        train.detach().cpu().numpy(), wavelet=wavelet, levels=levels
    )
    arg = experiment.argumentation_table(
        mask_dict,
        frame,
        train_power,
        mass=mass,
        energies=energies,
        noise_std=float(cfg.get("measurement", {}).get("noise_std", 0.0)),
    )
    arg.to_csv(run / "metrics" / "argumentation.csv", index=False)
    outcomes = [
        c
        for c in arg.columns
        if c.startswith(("complex_mse_", "magnitude_mse_", "psnr_gain_"))
    ]
    corr_parts = []
    for acquisition_family, family_table in arg.groupby("acquisition_family"):
        family_corr = experiment.rank_correlations(
            family_table, PREDICTORS, outcomes
        )
        family_corr.insert(0, "acquisition_family", acquisition_family)
        corr_parts.append(family_corr)
    corr = pd.concat(corr_parts, ignore_index=True)
    corr.to_csv(run / "metrics" / "argumentation_correlations.csv", index=False)

    # Plots: score vs error, PSF profile overlay, crop-and-zoom comparison.
    for acquisition_family, family_summary in summary.groupby(
        "acquisition_family"
    ):
        viz.plot_score_vs_error(
            family_summary["mask"].tolist(),
            family_summary["method"].tolist(),
            family_summary["mask_score"].tolist(),
            family_summary["complex_mse_mean"].tolist(),
            run / "plots" / f"score_vs_error_{acquisition_family}.png",
        )
    viz.plot_psf_profiles(mask_dict, run / "plots" / "psf_profiles.png")

    # A predeclared middle index avoids choosing a showcase from test outcomes.
    median_idx = int(test.shape[0] // 2)
    noise_std = float(cfg.get("measurement", {}).get("noise_std", 0.0))
    zoom_noise = None
    if noise_std > 0.0:
        zoom_noise = recon.sample_complex_noise_like(
            test[median_idx : median_idx + 1],
            noise_std,
            generator=torch.Generator().manual_seed(int(cfg["seed"]) + 702),
        )
    zoom_recons = {}
    for name, mask in mask_dict.items():
        recons = experiment.reconstruct_all(
            test[median_idx : median_idx + 1],
            mask,
            cfg,
            noise=zoom_noise,
            spectrum=prior_variance,
            prior_mean=prior_mean,
        )
        method = "wavelet_ista" if "wavelet_ista" in recons else "zero_filled"
        zoom_recons[name] = recons[method][0].abs().detach().cpu().numpy()
    viz.plot_zoom_comparison(
        test[median_idx].detach().cpu().numpy(),
        zoom_recons,
        run / "plots" / "zoom_comparison.png",
        title=f"test[{median_idx}], wavelet_ista",
    )

    print("\nresults table (summary.csv):")
    cols = [
        "acquisition_family",
        "mask",
        "method",
        "actual_acceleration",
        "psnr_mean",
        "ssim_mean",
        "nrmse_mean",
        "complex_mse_mean",
        "mask_score",
    ]
    print(summary[cols].round(4).to_string(index=False))
    print("\nargumentation table (argumentation.csv):")
    print(arg.round(4).to_string(index=False))
    print("\npredictor-outcome rank correlations:")
    print(corr.round(3).to_string(index=False))
    print(f"\noutputs: {run / 'metrics'}  |  {run / 'plots'}")


if __name__ == "__main__":
    main()
