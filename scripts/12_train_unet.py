#!/usr/bin/env python3
"""Train a U-Net post-processor as the learned-prior arm.

Input  = zero-filled magnitude of the train split under a fixed mask
Target = the reference signal
Loss   = MSE

The trained model is saved to runs/<experiment_name>/models/unet_post.pt and
picked up automatically by scripts/07 and scripts/11, which register it as
the "unet_post" reconstruction method. Two properties are expected and
reported rather than fixed:

- unet_post imputes null-space content (recon_nullspace_norm > 0), like
  wavelet_ista and subspace but unlike zero-filling and Wiener.
- unet_post enforces no data consistency (there is no analogue of ISTA's
  final_dc step), so its consistency_norm is nonzero. That is a property of
  post-processing, not a bug.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mrsim import experiment, viz
from mrsim.config import load_config, run_dir, save_config_snapshot, seed_everything
from mrsim.fft_ops import ifft2c
from mrsim.progress import track
from mrsim.recon import simulate_measurements
from mrsim.unet import SmallUNet

# SmallUNet pools twice, so both side lengths must be divisible by 4.
POOL_FACTOR = 4


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--mask-type", default=None, help="mask to train under")
    args = parser.parse_args()

    cfg = load_config(args.config)
    rng = seed_everything(int(cfg["seed"]))
    run = run_dir(cfg)
    save_config_snapshot(cfg, run, "12_train_unet")

    images = experiment.load_or_generate_dataset(cfg, run)
    train, test = experiment.train_test_split(images, cfg)
    height, width = train.shape[-2:]
    if height % POOL_FACTOR or width % POOL_FACTOR:
        raise ValueError(
            f"image size {height}x{width} must be divisible by {POOL_FACTOR} "
            "(the U-Net pools twice)"
        )

    unet_cfg = cfg.get("unet_post", {})
    mask_type = args.mask_type or str(unet_cfg.get("mask_type", "variable_density"))
    mask = experiment.build_masks([mask_type], cfg, train, rng)[mask_type]

    noise_std = float(cfg.get("measurement", {}).get("noise_std", 0.0))
    generator = torch.Generator().manual_seed(int(cfg["seed"]))

    def zero_filled_batch(batch: torch.Tensor) -> torch.Tensor:
        y = simulate_measurements(batch, mask, noise_std=noise_std, generator=generator)
        return ifft2c(y).abs()

    model = SmallUNet(base_channels=int(unet_cfg.get("base_channels", 16)))
    optimizer = torch.optim.Adam(model.parameters(), lr=float(unet_cfg.get("lr", 1e-3)))
    epochs = int(unet_cfg.get("epochs", 60))
    batch_size = int(unet_cfg.get("batch_size", 8))

    order = np.arange(train.shape[0])
    shuffle_rng = np.random.default_rng(int(cfg["seed"]))
    losses: list[float] = []
    model.train()
    for _ in track(range(epochs), total=epochs, label="unet training"):
        shuffle_rng.shuffle(order)
        epoch_loss = 0.0
        for start in range(0, len(order), batch_size):
            batch = train[order[start : start + batch_size]]
            inputs = zero_filled_batch(batch).unsqueeze(1)
            outputs = model(inputs).squeeze(1)
            loss = ((outputs - batch) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss) * batch.shape[0]
        losses.append(epoch_loss / len(order))

    models_dir = run / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "base_channels": int(unet_cfg.get("base_channels", 16)),
            "mask_type": mask_type,
            "image_size": [height, width],
        },
        models_dir / "unet_post.pt",
    )
    np.save(models_dir / "unet_train_mask.npy", mask)
    viz.plot_series(
        losses, run / "plots" / "unet_loss.png",
        xlabel="epoch", ylabel="train MSE", title=f"U-Net post-processor ({mask_type})",
        logy=True,
    )

    # Held-out check on the same mask the model was trained under.
    spectrum = experiment.mean_power_spectrum(train)
    frame = experiment.evaluate_masks(
        {mask_type: mask}, test, cfg, run,
        prefix="unet_post", spectrum=spectrum, unet_model=model,
    )
    summary = frame.groupby("method")[
        ["psnr", "ssim", "nrmse", "recon_nullspace_norm", "consistency_norm"]
    ].mean()
    pd.DataFrame([{"final_train_mse": losses[-1], "mask_type": mask_type}]).to_csv(
        run / "metrics" / "unet_train.csv", index=False
    )

    print(f"\ntrained under mask '{mask_type}'; final train MSE {losses[-1]:.5f}")
    print(summary.round(4).to_string())
    print(f"\nmodel: {models_dir / 'unet_post.pt'}")


if __name__ == "__main__":
    main()
