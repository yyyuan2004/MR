#!/usr/bin/env python3
"""Train a U-Net post-processor as the learned-prior arm.

Input  = zero-filled magnitude of the train split under a fixed mask
Target = the reference signal
Loss   = MSE

The trained model is saved below the config-fingerprinted run directory.
Scripts 07 and 11 include it only when
``experimental.include_unet_in_comparisons`` is enabled. Two properties are
expected and reported rather than fixed:

- unet_post imputes null-space content (recon_nullspace_norm > 0), like
  wavelet_ISTA, affine subspace reconstruction, and a nonzero-mean Wiener
  estimator, but unlike zero filling.
- unet_post enforces no hard data consistency (there is no analogue of ISTA's
  final_dc step), so its observable measurement residual can be nonzero. That
  is a reported property of post-processing, not a test-time certificate.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mrsim import experiment, recon, viz
from mrsim.config import (
    config_fingerprint,
    load_config,
    run_dir,
    save_config_snapshot,
    seed_everything,
)
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
    save_config_snapshot(cfg, run, "12_train_unet", cli_args=vars(args))

    images = experiment.load_or_generate_dataset(cfg, run)
    train, validation, test = experiment.train_validation_test_split(images, cfg)
    if validation.shape[0] == 0:
        raise ValueError("U-Net checkpoint selection requires data.n_val > 0")
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
    train_noise = None
    validation_noise = None
    if noise_std > 0.0:
        train_noise = recon.sample_complex_noise_like(
            train,
            noise_std,
            generator=torch.Generator().manual_seed(int(cfg["seed"]) + 1201),
        )
        validation_noise = recon.sample_complex_noise_like(
            validation,
            noise_std,
            generator=torch.Generator().manual_seed(int(cfg["seed"]) + 1202),
        )

    def zero_filled_batch(
        batch: torch.Tensor, explicit_noise: torch.Tensor | None
    ) -> torch.Tensor:
        if explicit_noise is None:
            y = simulate_measurements(batch, mask)
        else:
            y = simulate_measurements(batch, mask, noise=explicit_noise)
        return ifft2c(y).abs()

    model = SmallUNet(base_channels=int(unet_cfg.get("base_channels", 16)))
    optimizer = torch.optim.Adam(model.parameters(), lr=float(unet_cfg.get("lr", 1e-3)))
    epochs = int(unet_cfg.get("epochs", 60))
    batch_size = int(unet_cfg.get("batch_size", 8))

    order = np.arange(train.shape[0])
    shuffle_rng = np.random.default_rng(int(cfg["seed"]))
    train_losses: list[float] = []
    validation_losses: list[float] = []
    best_validation = float("inf")
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in track(range(epochs), total=epochs, label="unet training"):
        model.train()
        shuffle_rng.shuffle(order)
        epoch_loss = 0.0
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            batch = train[indices]
            batch_noise = None if train_noise is None else train_noise[indices]
            inputs = zero_filled_batch(batch, batch_noise).unsqueeze(1)
            outputs = model(inputs).squeeze(1)
            loss = ((outputs - batch) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss) * batch.shape[0]
        train_losses.append(epoch_loss / len(order))

        model.eval()
        with torch.no_grad():
            validation_inputs = zero_filled_batch(
                validation, validation_noise
            ).unsqueeze(1)
            validation_outputs = model(validation_inputs).squeeze(1)
            validation_loss = float(((validation_outputs - validation) ** 2).mean())
        validation_losses.append(validation_loss)
        if validation_loss < best_validation:
            best_validation = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_state)

    models_dir = run / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "base_channels": int(unet_cfg.get("base_channels", 16)),
            "mask_type": mask_type,
            "image_size": [height, width],
            "config_fingerprint": config_fingerprint(cfg),
            "mask_sha256": hashlib.sha256(
                np.ascontiguousarray(mask).tobytes()
            ).hexdigest(),
            "best_epoch": best_epoch,
            "best_validation_mse": best_validation,
        },
        models_dir / "unet_post.pt",
    )
    np.save(models_dir / "unet_train_mask.npy", mask)
    viz.plot_series(
        validation_losses, run / "plots" / "unet_validation_loss.png",
        xlabel="epoch",
        ylabel="validation MSE",
        title=f"U-Net validation curve ({mask_type})",
        logy=True,
    )

    # Held-out check on the same mask the model was trained under.
    prior_mean, prior_variance, _ = experiment.frequency_prior_statistics(train)
    frame = experiment.evaluate_masks(
        {mask_type: mask}, test, cfg, run,
        prefix="unet_post",
        spectrum=prior_variance,
        prior_mean=prior_mean,
        unet_model=model,
    )
    summary = frame.groupby("method")[
        [
            "psnr",
            "ssim",
            "nrmse",
            "recon_nullspace_norm",
            "measurement_residual_norm",
        ]
    ].mean()
    pd.DataFrame(
        [
            {
                "best_epoch": best_epoch,
                "best_validation_mse": best_validation,
                "final_train_mse": train_losses[-1],
                "mask_type": mask_type,
            }
        ]
    ).to_csv(
        run / "metrics" / "unet_train.csv", index=False
    )

    print(
        f"\ntrained under mask '{mask_type}'; selected epoch {best_epoch + 1} "
        f"with validation MSE {best_validation:.5f}"
    )
    print(summary.round(4).to_string())
    print(f"\nmodel: {models_dir / 'unet_post.pt'}")


if __name__ == "__main__":
    main()
