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
import copy
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
        raw = torch.sigmoid(self.prob_slope * self.logits)
        n_total = raw.numel()
        n_center = int(self.center.sum())
        n_free = n_total - n_center
        target_total = self.target_fraction * n_total
        target_free = target_total - n_center
        if n_free <= 0 or not 0.0 <= target_free <= n_free:
            raise ValueError(
                "the forced center lines are incompatible with the requested sampling budget"
            )

        free = raw[~self.center]
        target_mean = free.new_tensor(target_free / n_free)
        current_mean = free.mean().clamp(1e-6, 1.0 - 1e-6)
        scaled_down = free * (target_mean / current_mean)
        scaled_up = 1.0 - (1.0 - free) * ((1.0 - target_mean) / (1.0 - current_mean))
        projected = torch.where(current_mean >= target_mean, scaled_down, scaled_up)
        out = torch.ones_like(raw)
        out[~self.center] = projected.clamp(0.0, 1.0)
        return out

    def sample(self, batch_size: int) -> torch.Tensor:
        p = self.probabilities()
        u = torch.rand(batch_size, p.numel(), device=p.device)
        soft = torch.sigmoid(self.mask_slope * (p - u))
        hard = (p.unsqueeze(0) >= u).to(soft.dtype)
        # Hard forward pass, soft gradient (straight-through relaxation).
        return hard + soft - soft.detach()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(int(cfg["seed"]))
    run = run_dir(cfg)
    save_config_snapshot(cfg, run, "09_loupe_baseline", cli_args=vars(args))

    images = experiment.load_or_generate_dataset(cfg, run)
    train, validation, test = experiment.train_validation_test_split(images, cfg)
    if validation.shape[0] == 0:
        raise ValueError("LOUPE checkpoint selection requires data.n_val > 0")
    shape, n_samples, _ = experiment.mask_budgets(cfg)
    n_rows, n_cols = shape
    if n_rows % 4 or n_cols % 4:
        raise ValueError(
            f"image size {n_rows}x{n_cols} must be divisible by 4 "
            "(the U-Net pools twice)"
        )
    n_lines = masks.line_count_from_sample_budget(shape, n_samples)

    loupe_cfg = cfg.get("loupe", {})
    n_center_lines = int(cfg.get("mask", {}).get("lines", {}).get("n_center_lines", 2))
    offsets = np.abs(np.arange(n_cols) - n_cols // 2)
    center_cols = np.argsort(offsets, kind="stable")[: min(n_center_lines, n_lines)]

    sampler = LoupeLineSampler(
        n_cols,
        target_fraction=float(n_lines / n_cols),
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
    validation_losses: list[float] = []
    best_validation = float("inf")
    best_state: dict | None = None
    order = np.arange(train.shape[0])
    rng = np.random.default_rng(int(cfg["seed"]))
    noise_std = float(cfg.get("measurement", {}).get("noise_std", 0.0))
    validation_noise = None
    if noise_std > 0.0:
        from mrsim.recon import sample_complex_noise_like

        validation_noise = sample_complex_noise_like(
            validation,
            noise_std,
            generator=torch.Generator().manual_seed(int(cfg["seed"]) + 902),
        )
    for _ in track(range(epochs), total=epochs, label="loupe training"):
        sampler.train()
        unet.train()
        rng.shuffle(order)
        epoch_loss = 0.0
        for start in range(0, len(order), batch_size):
            batch = train[order[start : start + batch_size]]
            k = fft2c(batch.to(torch.complex64))
            if noise_std > 0.0:
                noise = (noise_std / np.sqrt(2.0)) * (
                    torch.randn_like(k.real) + 1j * torch.randn_like(k.real)
                )
                k = k + noise
            line_mask = sampler.sample(batch.shape[0]).unsqueeze(1)  # (B, 1, W)
            zero_filled = ifft2c(k * line_mask).abs()
            recon = unet(zero_filled.unsqueeze(1)).squeeze(1)
            loss = ((recon - batch) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss) * batch.shape[0]
        losses.append(epoch_loss / len(order))

        sampler.eval()
        unet.eval()
        with torch.no_grad():
            probabilities = sampler.probabilities().cpu().numpy()
            free = np.setdiff1d(np.arange(n_cols), center_cols)
            free = free[np.argsort(probabilities[free], kind="stable")[::-1]]
            hard_mask = masks.fill_full_lines(
                shape,
                np.concatenate([center_cols, free]),
                n_lines,
            )
            val_k = fft2c(validation.to(torch.complex64))
            if validation_noise is not None:
                val_k = val_k + validation_noise
            hard_mask_tensor = torch.from_numpy(hard_mask).to(val_k)
            val_zf = ifft2c(val_k * hard_mask_tensor).abs()
            val_recon = unet(val_zf.unsqueeze(1)).squeeze(1)
            val_loss = float(((val_recon - validation) ** 2).mean())
        validation_losses.append(val_loss)
        if val_loss < best_validation:
            best_validation = val_loss
            best_state = {
                "sampler": copy.deepcopy(sampler.state_dict()),
                "unet": copy.deepcopy(unet.state_dict()),
            }

    if best_state is not None:
        sampler.load_state_dict(best_state["sampler"])
        unet.load_state_dict(best_state["unet"])

    # Binarize: forced center columns first, then columns by learned probability.
    with torch.no_grad():
        probs = sampler.probabilities().detach().cpu().numpy()
    rest = np.setdiff1d(np.arange(n_cols), center_cols)
    rest = rest[np.argsort(probs[rest], kind="stable")[::-1]]
    mask = masks.fill_full_lines(
        shape, np.concatenate([center_cols, rest]), n_lines
    )

    experiment.save_mask_bundle(mask, "loupe_learned", run)
    diagnostics = run / "diagnostics"
    diagnostics.mkdir(parents=True, exist_ok=True)
    np.save(diagnostics / "loupe_probabilities.npy", probs)
    viz.plot_series(losses, run / "plots" / "loupe_loss.png",
                    xlabel="epoch", ylabel="train MSE", title="learned-mask training loss")
    viz.plot_series(probs, run / "plots" / "loupe_probabilities.png",
                    xlabel="column index", ylabel="probability", title="learned column probabilities")
    if validation_losses:
        viz.plot_series(
            validation_losses,
            run / "plots" / "loupe_validation_loss.png",
            xlabel="epoch",
            ylabel="validation MSE",
            title="hard-mask validation loss",
        )

    # Evaluate the binarized mask with the shared methods.
    prior_mean, prior_variance, _ = experiment.frequency_prior_statistics(train)
    frame = experiment.evaluate_masks(
        {"loupe_learned": mask},
        test,
        cfg,
        run,
        prefix="loupe",
        spectrum=prior_variance,
        prior_mean=prior_mean,
    )
    summary = frame.groupby("method")[["psnr", "ssim", "nrmse"]].mean()

    # The trained network's own test error, on the binarized mask.
    unet.eval()
    with torch.no_grad():
        from mrsim.recon import simulate_measurements

        test_noise = None
        if noise_std > 0.0:
            from mrsim.recon import sample_complex_noise_like

            test_noise = sample_complex_noise_like(
                test,
                noise_std,
                generator=torch.Generator().manual_seed(int(cfg["seed"]) + 701),
            )
        y = simulate_measurements(test, mask, noise=test_noise)
        zero_filled = ifft2c(y).abs()
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
