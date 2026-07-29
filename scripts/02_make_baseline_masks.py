#!/usr/bin/env python3
"""Build the stable point and full-line baselines and evaluate the test split."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mrsim import experiment
from mrsim.config import load_config, run_dir, save_config_snapshot, seed_everything

BASELINES = [
    "uniform_random",
    "variable_density",
    "multilevel_random",
    "equispaced_lines",
    "variable_density_lines",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    rng = seed_everything(int(cfg["seed"]))
    run = run_dir(cfg)
    save_config_snapshot(cfg, run, "02_make_baseline_masks", cli_args=vars(args))

    images = experiment.load_or_generate_dataset(cfg, run)
    train, _, test = experiment.train_validation_test_split(images, cfg)

    mask_dict = experiment.build_masks(BASELINES, cfg, train, rng)
    prior_mean, prior_variance, _ = experiment.frequency_prior_statistics(train)
    frame = experiment.evaluate_masks(
        mask_dict,
        test,
        cfg,
        run,
        prefix="baselines",
        spectrum=prior_variance,
        prior_mean=prior_mean,
    )

    summary = frame.groupby(["mask", "method"])[["psnr", "ssim", "nrmse"]].mean()
    print(summary.round(4).to_string())
    print(f"\nmetrics: {run / 'metrics' / 'baselines_metrics.csv'}")


if __name__ == "__main__":
    main()
