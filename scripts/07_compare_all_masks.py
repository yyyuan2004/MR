#!/usr/bin/env python3
"""Default experiment: build every configured mask, reconstruct the test split
with zero-filled and ridge methods, and compare masks.

Outputs (under runs/<experiment_name>/):
  metrics/compare_metrics.csv       per-image metrics for every mask x method
  metrics/compare_psf_metrics.csv   PSF metrics per mask
  metrics/summary.csv               aggregated comparison table
  plots/score_vs_error.png          mask score vs true reconstruction error
  masks/, psf/, recon/, artifact_maps/  per-mask images
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mrsim import artifacts, experiment, viz
from mrsim.config import load_config, run_dir, save_config_snapshot, seed_everything


def representative_indices(frame: pd.DataFrame, n_examples: int, n_test: int) -> list[int]:
    """Test indices spread across the difficulty range (quantiles of mean NRMSE)."""
    per_image = frame.groupby("image_index")["nrmse"].mean().reindex(range(n_test))
    order = per_image.sort_values().index.to_numpy()
    positions = np.unique(np.linspace(0, len(order) - 1, n_examples).astype(int))
    return sorted(int(order[p]) for p in positions)


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

    # First pass to find representative examples, then evaluate with them.
    n_examples = int(cfg.get("outputs", {}).get("n_examples", 5))
    frame = experiment.evaluate_masks(mask_dict, test, cfg, run, prefix="compare")
    examples = representative_indices(frame, n_examples, test.shape[0])
    frame = experiment.evaluate_masks(
        mask_dict, test, cfg, run, prefix="compare", example_indices=examples
    )

    # Mask score: expected zero-filled MSE under the train mean power spectrum.
    train_power = experiment.mean_power_spectrum(train)
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

    # Scatter: predicted mask score vs measured mean reconstruction MSE.
    points = summary[["mask", "method", "mask_score", "mse_mean"]]
    viz.scatter_with_labels(
        points["mask_score"].tolist(),
        points["mse_mean"].tolist(),
        [f"{m}/{r}" for m, r in zip(points["mask"], points["method"])],
        run / "plots" / "score_vs_error.png",
        xlabel="mask score (expected zero-filled MSE, train spectrum)",
        ylabel="measured mean MSE (test split)",
        title="mask score vs true reconstruction error",
    )

    print(f"\nrepresentative test indices: {examples}")
    cols = ["mask", "method", "psnr_mean", "ssim_mean", "nrmse_mean", "mask_score"]
    print(summary[cols].round(4).to_string(index=False))
    print(f"\nsummary: {run / 'metrics' / 'summary.csv'}")
    print(f"scatter: {run / 'plots' / 'score_vs_error.png'}")


if __name__ == "__main__":
    main()
