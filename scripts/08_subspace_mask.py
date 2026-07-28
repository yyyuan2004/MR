#!/usr/bin/env python3
"""Subspace-prior sampling design: fit a linear subspace on the train split,
select a mask by subspace A-optimal greedy (Sherman-Morrison accelerated),
reconstruct with the closed-form subspace method, and evaluate with the full
error decomposition.

Note: the subspace reconstruction lives in the span of the fitted basis, not
in the observed subspace, so its recon_nullspace_norm is genuinely nonzero —
a linear method that imputes null-space content.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mrsim import artifacts, experiment, greedy, subspace, viz
from mrsim.config import load_config, run_dir, save_config_snapshot, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(int(cfg["seed"]))
    run = run_dir(cfg)
    save_config_snapshot(cfg, run, "08_subspace_mask")

    images = experiment.load_or_generate_dataset(cfg, run)
    train, test = experiment.train_test_split(images, cfg)

    basis, energy_ratio = experiment.fit_train_subspace(cfg, train)
    phi = subspace.to_kspace_basis(basis, tuple(train.shape[-2:]))
    shape, n_samples, n_center = experiment.mask_budgets(cfg)
    sub_cfg = cfg.get("subspace", {})

    mask, trace_history = greedy.greedy_subspace_aoptimal(
        phi,
        n_samples,
        sigma2=experiment.greedy_noise_var(cfg),
        n_center=n_center,
        beta=float(sub_cfg.get("beta", 0.0)),
        shape=shape,
        ridge=float(sub_cfg.get("ridge", 1e-6)),
        n_candidates=int(cfg.get("greedy", {}).get("n_candidates", 32)),
        return_trace=True,
    )
    viz.plot_series(
        trace_history,
        run / "plots" / "subspace_trace.png",
        xlabel="greedy step",
        ylabel="sigma^2 * trace(J^-1)",
        title="subspace A-optimal design criterion",
        logy=True,
    )
    leakage = artifacts.subspace_nullspace_leakage(basis, mask)

    spectrum = experiment.mean_power_spectrum(train)
    frame = experiment.evaluate_masks(
        {"subspace_aopt_greedy": mask}, test, cfg, run,
        prefix="subspace", spectrum=spectrum, subspace_basis=basis,
    )

    print(f"\nsubspace d={basis.shape[1]}: captured train energy {energy_ratio:.4f}")
    print(f"design criterion: {trace_history[0]:.4g} -> {trace_history[-1]:.4g}")
    print(f"subspace null-space leakage of the mask: {leakage:.4f}")
    summary = frame.groupby("method")[["psnr", "ssim", "nrmse", "recon_nullspace_norm"]].mean()
    print(summary.round(4).to_string())
    print(f"\nmetrics: {run / 'metrics' / 'subspace_metrics.csv'}")
    print(f"trace plot: {run / 'plots' / 'subspace_trace.png'}")


if __name__ == "__main__":
    main()
