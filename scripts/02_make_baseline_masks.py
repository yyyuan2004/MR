#!/usr/bin/env python3
"""Build the baseline masks (uniform random, variable density, equispaced lines)
and evaluate them on the test split."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mrsim import experiment
from mrsim.config import load_config, run_dir, save_config_snapshot, seed_everything

BASELINES = ["uniform_random", "variable_density", "equispaced_lines"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    rng = seed_everything(int(cfg["seed"]))
    run = run_dir(cfg)
    save_config_snapshot(cfg, run, "02_make_baseline_masks")

    images = experiment.load_or_generate_dataset(cfg, run)
    train, test = experiment.train_test_split(images, cfg)

    mask_dict = experiment.build_masks(BASELINES, cfg, train, rng)
    spectrum = experiment.mean_power_spectrum(train)
    frame = experiment.evaluate_masks(
        mask_dict, test, cfg, run, prefix="baselines", spectrum=spectrum
    )

    summary = frame.groupby(["mask", "method"])[["psnr", "ssim", "nrmse"]].mean()
    print(summary.round(4).to_string())
    print(f"\nmetrics: {run / 'metrics' / 'baselines_metrics.csv'}")


if __name__ == "__main__":
    main()
