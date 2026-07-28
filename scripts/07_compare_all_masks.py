#!/usr/bin/env python3
"""Default experiment: build every configured mask, reconstruct the test split
with zero-filled, Wiener, and wavelet-ISTA methods, and compare masks.

Outputs (under runs/<experiment_name>/):
  metrics/compare_metrics.csv        per-image metrics for every mask x method
  metrics/compare_psf_metrics.csv    PSF metrics per mask
  metrics/summary.csv                aggregated results table
  metrics/argumentation.csv          design-time scores vs measured outcomes
  metrics/argumentation_correlations.csv  Spearman rank correlations
  plots/score_vs_error.png           mask score vs measured error
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

from mrsim import artifacts, experiment, viz
from mrsim.config import load_config, run_dir, save_config_snapshot, seed_everything

PREDICTORS = [
    "mask_score",
    "wavelet_leakage",
    "weighted_max_sidelobe",
    "psf_max_sidelobe",
    "truth_nullspace_norm",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    rng = seed_everything(int(cfg["seed"]))
    run = run_dir(cfg)
    save_config_snapshot(cfg, run, "07_compare_all_masks")

    images = experiment.load_or_generate_dataset(cfg, run)
    train, test = experiment.train_test_split(images, cfg)
    mask_names = list(cfg["mask"]["types"])
    print(f"building masks: {', '.join(mask_names)}")
    mask_dict = experiment.build_masks(mask_names, cfg, train, rng)

    # Mean power spectrum of the train split: prior for wiener and mask scores.
    train_power = experiment.mean_power_spectrum(train)
    unet = experiment.load_unet(run)
    if unet is not None:
        print("including the trained U-Net post-processor as 'unet_post'")
    frame = experiment.evaluate_masks(
        mask_dict, test, cfg, run, prefix="compare",
        spectrum=train_power, unet_model=unet,
    )

    scores = {
        name: artifacts.expected_zero_filled_mse(mask, train_power)
        for name, mask in mask_dict.items()
    }
    summary = (
        frame.groupby(["mask", "method"])[["mse", "psnr", "ssim", "nrmse", "aliasing_energy_ratio"]]
        .agg(["mean", "std"])
    )
    summary.columns = ["_".join(col) for col in summary.columns]
    summary = summary.reset_index()
    summary["mask_score"] = summary["mask"].map(scores)
    psf_frame = pd.read_csv(run / "metrics" / "compare_psf_metrics.csv")
    summary = summary.merge(psf_frame, on="mask", how="left")
    summary.to_csv(run / "metrics" / "summary.csv", index=False)

    # Argumentation table: do the design-time scores predict the outcomes?
    ista_cfg = cfg.get("recon", {}).get("wavelet_ista", {})
    wavelet = str(ista_cfg.get("wavelet", "db4"))
    levels = int(ista_cfg.get("levels", 3))
    mass = artifacts.subband_spectral_mass(train_power.shape, wavelet=wavelet, levels=levels)
    energies = artifacts.subband_energies(train.numpy(), wavelet=wavelet, levels=levels)
    arg = experiment.argumentation_table(
        mask_dict, frame, train_power, mass=mass, energies=energies
    )
    arg.to_csv(run / "metrics" / "argumentation.csv", index=False)
    outcomes = [c for c in arg.columns if c.startswith(("mse_", "psnr_gain_"))]
    corr = experiment.rank_correlations(arg, PREDICTORS, outcomes)
    corr.to_csv(run / "metrics" / "argumentation_correlations.csv", index=False)

    # Plots: score vs error, PSF profile overlay, crop-and-zoom comparison.
    viz.plot_score_vs_error(
        summary["mask"].tolist(),
        summary["method"].tolist(),
        summary["mask_score"].tolist(),
        summary["mse_mean"].tolist(),
        run / "plots" / "score_vs_error.png",
    )
    viz.plot_psf_profiles(mask_dict, run / "plots" / "psf_profiles.png")

    # Zoom comparison on the median-difficulty test image, wavelet-ISTA method.
    median_idx = experiment.representative_indices(frame, 3, test.shape[0])[1]
    generator = torch.Generator().manual_seed(int(cfg["seed"]))
    zoom_recons = {}
    for name, mask in mask_dict.items():
        recons = experiment.reconstruct_all(
            test[median_idx : median_idx + 1], mask, cfg, generator=generator, spectrum=train_power
        )
        method = "wavelet_ista" if "wavelet_ista" in recons else "zero_filled"
        zoom_recons[name] = recons[method][0].abs().numpy()
    viz.plot_zoom_comparison(
        test[median_idx].numpy(),
        zoom_recons,
        run / "plots" / "zoom_comparison.png",
        title=f"test[{median_idx}], wavelet_ista",
    )

    print("\nresults table (summary.csv):")
    cols = ["mask", "method", "psnr_mean", "ssim_mean", "nrmse_mean", "mask_score"]
    print(summary[cols].round(4).to_string(index=False))
    print("\nargumentation table (argumentation.csv):")
    print(arg.round(4).to_string(index=False))
    print("\npredictor-outcome rank correlations:")
    print(corr.round(3).to_string(index=False))
    print(f"\noutputs: {run / 'metrics'}  |  {run / 'plots'}")


if __name__ == "__main__":
    main()
