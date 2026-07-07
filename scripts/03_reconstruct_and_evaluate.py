#!/usr/bin/env python3
"""Reconstruct the test split with every mask saved in the run directory
(runs/<experiment_name>/masks/*.npy) and write per-image metrics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mrsim import experiment
from mrsim.config import load_config, run_dir, save_config_snapshot, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    rng = seed_everything(int(cfg["seed"]))
    run = run_dir(cfg)
    save_config_snapshot(cfg, run, "03_reconstruct_and_evaluate")

    images = experiment.load_or_generate_dataset(cfg, run)
    train, test = experiment.train_test_split(images, cfg)

    mask_files = sorted((run / "masks").glob("*.npy")) if (run / "masks").exists() else []
    if mask_files:
        mask_dict = {path.stem: np.load(path) for path in mask_files}
    else:
        # No masks saved yet: fall back to the baseline masks.
        mask_dict = experiment.build_masks(
            ["uniform_random", "variable_density", "equispaced_lines"], cfg, train, rng
        )

    frame = experiment.evaluate_masks(mask_dict, test, cfg, run, prefix="recon")

    summary = frame.groupby(["mask", "method"])[["psnr", "ssim", "nrmse"]].mean()
    print(summary.round(4).to_string())
    print(f"\nmetrics: {run / 'metrics' / 'recon_metrics.csv'}")


if __name__ == "__main__":
    main()
