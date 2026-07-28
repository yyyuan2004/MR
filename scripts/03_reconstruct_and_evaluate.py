#!/usr/bin/env python3
"""Reconstruct the test split with masks declared by the run's manifest.

When no manifest exists, build the masks declared by ``mask.types``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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
    save_config_snapshot(cfg, run, "03_reconstruct_and_evaluate", cli_args=vars(args))

    images = experiment.load_or_generate_dataset(cfg, run)
    train, _, test = experiment.train_validation_test_split(images, cfg)

    manifest_path = run / "masks" / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as handle:
            manifest = json.load(handle)
        mask_dict = {}
        expected_shape = tuple(test.shape[-2:])
        for name, item in manifest.get("masks", {}).items():
            path = manifest_path.parent / item["path"]
            mask = np.load(path)
            digest = hashlib.sha256(np.ascontiguousarray(mask).tobytes()).hexdigest()
            if mask.ndim != 2 or tuple(mask.shape) != expected_shape:
                raise ValueError(f"manifest mask {name!r} has incompatible shape {mask.shape}")
            if not np.isin(mask, (0.0, 1.0)).all():
                raise ValueError(f"manifest mask {name!r} is not binary")
            if digest != item["sha256"]:
                raise ValueError(f"manifest hash mismatch for mask {name!r}")
            mask_dict[name] = mask
    else:
        # No explicit mask manifest: build exactly the masks declared by config.
        mask_dict = experiment.build_masks(
            list(cfg["mask"]["types"]), cfg, train, rng
        )

    prior_mean, prior_variance, _ = experiment.frequency_prior_statistics(train)
    frame = experiment.evaluate_masks(
        mask_dict,
        test,
        cfg,
        run,
        prefix="recon",
        spectrum=prior_variance,
        prior_mean=prior_mean,
    )

    summary = frame.groupby(["mask", "method"])[["psnr", "ssim", "nrmse"]].mean()
    print(summary.round(4).to_string())
    print(f"\nmetrics: {run / 'metrics' / 'recon_metrics.csv'}")


if __name__ == "__main__":
    main()
