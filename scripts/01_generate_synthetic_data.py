#!/usr/bin/env python3
"""Generate the synthetic dataset for an experiment run."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mrsim import experiment, viz
from mrsim.config import load_config, run_dir, save_config_snapshot, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(int(cfg["seed"]))
    run = run_dir(cfg)
    save_config_snapshot(cfg, run, "01_generate_synthetic_data")

    images = experiment.load_or_generate_dataset(cfg, run, force=True)
    train, test = experiment.train_test_split(images, cfg)

    preview = [images[i].numpy() for i in range(min(16, images.shape[0]))]
    viz.save_image_grid(
        preview,
        run / "data" / "preview.png",
        titles=[f"[{i}]" for i in range(len(preview))],
        n_cols=4,
        vmin=0.0,
        vmax=1.0,
    )

    print(f"dataset: {experiment.dataset_path(run)}  shape={tuple(images.shape)}")
    print(f"split:   train={train.shape[0]}  test={test.shape[0]}")
    print(f"preview: {run / 'data' / 'preview.png'}")


if __name__ == "__main__":
    main()
