#!/usr/bin/env python3
"""LOUPE-style learned-mask baseline: jointly train a differentiable
probabilistic Cartesian line mask and a U-Net reconstructor, then binarize
the learned probabilities into an exact-budget mask (forced center columns).

The learned mask is saved as masks/loupe_learned.npy so that
scripts/10_compare_manifold_vs_learned.py can compare it against the
model-based designs. The trained U-Net's test error is reported separately —
the mask comparison itself uses the shared reconstruction methods.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mrsim import experiment, masks
from mrsim.config import load_config, run_dir, save_config_snapshot, seed_everything
from mrsim.fft_ops import fft2c, ifft2c
from mrsim.progress import track
from mrsim.unet import SmallUNet
from mrsim import viz


class LoupeLineSampler(nn.Module):
    """Differentiable probabilistic Cartesian line mask.

    Column probabilities come from learnable logits, rescaled so their mean
    matches the target column fraction (the differentiable budget constraint)
    and clamped to [0, 1]; forced center columns are pinned to 1. Sampling
    uses a sigmoid relaxation of the Bernoulli draw so gradients flow to the
    logits.
    """

    def __init__(
        self,
        n_cols: int,
        target_fraction: float,
        center_cols: np.ndarray,
        prob_slope: float = 5.0,
        mask_slope: float = 12.0,
    ):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(n_cols))
        self.target_fraction = target_fraction
        self.prob_slope = prob_slope
        self.mask_slope = mask_slope
        center = torch.zeros(n_cols, dtype=torch.bool)
        center[torch.from_numpy(center_cols.astype(np.int64))] = True
        self.register_buffer("center", center)

    def probabilities(self) -> torch.Tensor:
        p = torch.sigmoid(self.prob_slope * self.logits)
        p = p * (self.target_fraction / p.mean().clamp_min(1e-6))
        p = p.clamp(0.0, 1.0)
        return torch.where(self.center, torch.ones_like(p), p)

    def sample(self, batch_size: int) -> torch.Tensor:
        p = self.probabilities()
        u = torch.rand(batch_size, p.numel())
        return torch.sigmoid(self.mask_slope * (p - u))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(int(cfg["seed"]))
    run = run_dir(cfg)
    save_config_snapshot(cfg, run, "09_loupe_baseline")

    images = experiment.load_or_generate_dataset(cfg, run)
    train, test = experiment.train_test_split(images, cfg)
    shape, n_samples, _ = experiment.mask_budgets(cfg)
    n_rows, n_cols = shape

    loupe_cfg = cfg.get("loupe", {})
    n_center_lines = int(cfg.get("mask", {}).get("lines", {}).get("n_center_lines", 2))
    offsets = np.abs(np.arange(n_cols) - n_cols // 2)
    center_cols = np.argsort(offsets, kind="stable")[:n_center_lines]

    sampler = LoupeLineSampler(
        n_cols,
        target_fraction=float(cfg["mask"]["sampling_fraction"]),
        center_cols=center_cols,
        prob_slope=float(loupe_cfg.get("prob_slope", 5.0)),
        mask_slope=float(loupe_cfg.get("mask_slope", 12.0)),
    )
    unet = SmallUNet(base_channels=int(loupe_cfg.get("base_channels", 16)))
    optimizer = torch.optim.Adam(
        list(sampler.parameters()) + list(unet.parameters()),
        lr=float(loupe_cfg.get("lr", 1e-3)),
    )

    epochs = int(loupe_cfg.get("epochs", 30))
    batch_size = int(loupe_cfg.get("batch_size", 8))
    losses: list[float] = []
    order = np.arange(train.shape[0])
    rng = np.random.default_rng(int(cfg["seed"]))
    for _ in track(range(epochs), total=epochs, label="loupe training"):
        rng.shuffle(order)
        epoch_loss = 0.0
        for start in range(0, len(order), batch_size):
            batch = train[order[start : start + batch_size]]
            k = fft2c(batch.to(torch.complex64))
            line_mask = sampler.sample(batch.shape[0]).unsqueeze(1)  # (B, 1, W)
            zero_filled = ifft2c(k * line_mask).abs()
            recon = unet(zero_filled.unsqueeze(1)).squeeze(1)
            loss = ((recon - batch) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss) * batch.shape[0]
        losses.append(epoch_loss / len(order))

    # Binarize: forced center columns first, then columns by learned probability.
    with torch.no_grad():
        probs = sampler.probabilities().numpy()
    rest = np.setdiff1d(np.arange(n_cols), center_cols)
    rest = rest[np.argsort(probs[rest], kind="stable")[::-1]]
    mask = masks.fill_lines(shape, np.concatenate([center_cols, rest]), n_samples)

    experiment.save_mask_bundle(mask, "loupe_learned", run)
    np.save(run / "masks" / "loupe_probabilities.npy", probs)
    viz.plot_series(losses, run / "plots" / "loupe_loss.png",
                    xlabel="epoch", ylabel="train MSE", title="learned-mask training loss")
    viz.plot_series(probs, run / "plots" / "loupe_probabilities.png",
                    xlabel="column index", ylabel="probability", title="learned column probabilities")

    # Evaluate the binarized mask with the shared methods.
    spectrum = experiment.mean_power_spectrum(train)
    frame = experiment.evaluate_masks(
        {"loupe_learned": mask}, test, cfg, run, prefix="loupe", spectrum=spectrum
    )
    summary = frame.groupby("method")[["psnr", "ssim", "nrmse"]].mean()

    # The trained network's own test error, on the binarized mask.
    with torch.no_grad():
        k = fft2c(test.to(torch.complex64))
        zero_filled = ifft2c(k * torch.from_numpy(mask)).abs()
        recon = unet(zero_filled.unsqueeze(1)).squeeze(1)
        unet_mse = float(((recon - test) ** 2).mean())
        unet_psnr = float(10.0 * np.log10(1.0 / max(unet_mse, 1e-12)))
    pd.DataFrame(
        [{"final_train_mse": losses[-1], "unet_test_mse": unet_mse, "unet_test_psnr": unet_psnr}]
    ).to_csv(run / "metrics" / "loupe_unet.csv", index=False)

    print(f"\nfinal train MSE {losses[-1]:.5f}; trained-network test PSNR {unet_psnr:.2f} dB")
    print(summary.round(4).to_string())
    print(f"\nlearned mask: {run / 'masks' / 'loupe_learned.npy'}")


if __name__ == "__main__":
    main()
