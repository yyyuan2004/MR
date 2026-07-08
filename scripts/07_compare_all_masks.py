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
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mrsim import artifacts, experiment, viz
from mrsim.config import load_config, run_dir, save_config_snapshot, seed_everything

PREDICTORS = ["mask_score", "wavelet_leakage", "weighted_max_sidelobe", "psf_max_sidelobe"]
OUTCOMES = ["mse_zero_filled", "mse_wavelet_ista", "psnr_gain_ista"]


def argumentation_table(
    mask_dict: dict,
    frame: pd.DataFrame,
    train_power,
    mass: dict,
    energies: dict,
) -> pd.DataFrame:
    """Design-time scores next to measured outcomes, one row per mask.

    The point of this table is argumentative, not just descriptive: each
    predictor claims to rank masks before any data are measured, and the
    outcome columns test that claim (see the companion correlation table).
    """
    rows = []
    for name, mask in mask_dict.items():
        sub = frame[frame["mask"] == name]
        zf = sub[sub["method"] == "zero_filled"]
        ista = sub[sub["method"] == "wavelet_ista"]
        row: dict = {"mask": name}
        row["mask_score"] = artifacts.expected_zero_filled_mse(mask, train_power)
        row["wavelet_leakage"] = artifacts.wavelet_leakage_score(mask, mass, energies)
        row.update(artifacts.spectrum_weighted_psf_metrics(mask, train_power))
        row.update(artifacts.psf_metrics(mask))
        row["mse_zero_filled"] = float(zf["mse"].mean())
        if len(ista):
            row["mse_wavelet_ista"] = float(ista["mse"].mean())
            row["psnr_gain_ista"] = float(ista["psnr"].mean() - zf["psnr"].mean())
            row["ista_nullspace_norm"] = float(ista["recon_nullspace_norm"].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def correlation_table(arg: pd.DataFrame) -> pd.DataFrame:
    """Spearman rank correlation of each design-time predictor with each outcome."""
    rows = []
    for predictor in PREDICTORS:
        for outcome in OUTCOMES:
            if outcome not in arg or arg[outcome].isna().all():
                continue
            rho, p_value = spearmanr(arg[predictor], arg[outcome])
            rows.append(
                {
                    "predictor": predictor,
                    "outcome": outcome,
                    "spearman_rho": float(rho),
                    "p_value": float(p_value),
                }
            )
    return pd.DataFrame(rows)


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
    frame = experiment.evaluate_masks(mask_dict, test, cfg, run, prefix="compare", spectrum=train_power)

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
    arg = argumentation_table(mask_dict, frame, train_power, mass, energies)
    arg.to_csv(run / "metrics" / "argumentation.csv", index=False)
    corr = correlation_table(arg)
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
