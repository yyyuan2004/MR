#!/usr/bin/env python3
"""Build the artifact-aware greedy mask (expected-MSE gain traded against the
PSF max sidelobe) and evaluate it on the test split."""

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
    save_config_snapshot(cfg, run, "05_artifact_aware_mask_search")

    images = experiment.load_or_generate_dataset(cfg, run)
    train, test = experiment.train_test_split(images, cfg)

    mask_dict = experiment.build_masks(["artifact_aware_greedy"], cfg, train, rng)
    frame = experiment.evaluate_masks(mask_dict, test, cfg, run, prefix="artifact_aware")

    summary = frame.groupby(["mask", "method"])[["psnr", "ssim", "nrmse"]].mean()
    print(summary.round(4).to_string())
    print(f"\nmetrics: {run / 'metrics' / 'artifact_aware_metrics.csv'}")


if __name__ == "__main__":
    main()
