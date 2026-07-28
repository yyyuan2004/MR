#!/usr/bin/env python3
"""Subspace-prior sampling design: fit a linear subspace on the train split,
select a mask by subspace A-optimal greedy (Sherman-Morrison accelerated),
reconstruct with the closed-form subspace method, and evaluate with the full
error decomposition.

The PCA eigenvalues scale the covariance factor used consistently by design
and reconstruction. The estimate lives in a fitted affine subspace, not the
observed subspace, so it can impute null-space content.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

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
    save_config_snapshot(cfg, run, "08_subspace_mask", cli_args=vars(args))

    images = experiment.load_or_generate_dataset(cfg, run)
    train, _, test = experiment.train_validation_test_split(images, cfg)

    statistics = experiment.fit_train_subspace_statistics(cfg, train)
    basis = statistics.basis
    energy_ratio = statistics.energy_ratio
    phi = subspace.to_kspace_basis(basis, tuple(train.shape[-2:]))
    shape, n_samples, n_center = experiment.mask_budgets(cfg)
    sub_cfg = cfg.get("subspace", {})

    mask, trace_history = greedy.greedy_subspace_aoptimal(
        phi,
        n_samples,
        sigma2=experiment.subspace_regularization(cfg),
        prior_variances=statistics.eigenvalues,
        n_center=n_center,
        beta=float(sub_cfg.get("beta", 0.0)),
        shape=shape,
        ridge=experiment.subspace_regularization(cfg),
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
    leakage = artifacts.subspace_nullspace_leakage(
        basis, mask, weights=np.sqrt(statistics.eigenvalues)
    )

    prior_mean, prior_variance, _ = experiment.frequency_prior_statistics(train)
    frame = experiment.evaluate_masks(
        {"subspace_aopt_greedy": mask}, test, cfg, run,
        prefix="subspace",
        spectrum=prior_variance,
        prior_mean=prior_mean,
        subspace_basis=basis,
        subspace_mean=statistics.mean_image,
        subspace_variances=statistics.eigenvalues,
    )

    print(
        f"\nsubspace d={basis.shape[1]}: "
        f"explained centered train variance {energy_ratio:.4f}"
    )
    print(f"design criterion: {trace_history[0]:.4g} -> {trace_history[-1]:.4g}")
    print(f"subspace null-space leakage of the mask: {leakage:.4f}")
    summary = frame.groupby("method")[["psnr", "ssim", "nrmse", "recon_nullspace_norm"]].mean()
    print(summary.round(4).to_string())
    print(f"\nmetrics: {run / 'metrics' / 'subspace_metrics.csv'}")
    print(f"trace plot: {run / 'plots' / 'subspace_trace.png'}")


if __name__ == "__main__":
    main()
