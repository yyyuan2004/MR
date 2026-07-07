#!/usr/bin/env python3
"""Build the data-driven greedy mask from the empirical spectral energy of the
train split and evaluate it on the test split."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

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
    save_config_snapshot(cfg, run, "06_greedy_data_driven_mask")

    images = experiment.load_or_generate_dataset(cfg, run)
    train, test = experiment.train_test_split(images, cfg)

    mask_dict = experiment.build_masks(["data_driven_greedy"], cfg, train, rng)
    spectrum = experiment.mean_power_spectrum(train)
    frame = experiment.evaluate_masks(
        mask_dict, test, cfg, run, prefix="data_driven", spectrum=spectrum
    )

    summary = frame.groupby(["mask", "method"])[["psnr", "ssim", "nrmse"]].mean()
    print(summary.round(4).to_string())
    print(f"\nmetrics: {run / 'metrics' / 'data_driven_metrics.csv'}")


if __name__ == "__main__":
    main()
