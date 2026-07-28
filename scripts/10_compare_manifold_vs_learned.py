#!/usr/bin/env python3
"""Compare model-based sampling designs against the learned-mask baseline:
subspace A-optimal (manifold prior), sparse diagonal-prior A-optimal,
variable density, and the learned probabilistic line mask from script 09
(if it has been trained).

Reports, per mask: the subspace null-space leakage (the manifold coherence
metric), the spectral mask score, and measured errors for every shared
reconstruction method — then Spearman rank correlations and a
predicted-vs-measured scatter, focusing on whether the manifold metric
predicts (and ranks) the learned mask.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mrsim import artifacts, experiment, viz
from mrsim.config import load_config, run_dir, save_config_snapshot, seed_everything

MODEL_BASED = ["variable_density", "aopt_greedy", "subspace_aopt_greedy"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    rng = seed_everything(int(cfg["seed"]))
    run = run_dir(cfg)
    save_config_snapshot(cfg, run, "10_compare_manifold_vs_learned")

    images = experiment.load_or_generate_dataset(cfg, run)
    train, test = experiment.train_test_split(images, cfg)

    print(f"building masks: {', '.join(MODEL_BASED)}")
    mask_dict = experiment.build_masks(MODEL_BASED, cfg, train, rng)
    loupe_path = run / "masks" / "loupe_learned.npy"
    if loupe_path.exists():
        mask_dict["loupe_learned"] = np.load(loupe_path)
    else:
        print(f"no learned mask at {loupe_path}; run scripts/09_loupe_baseline.py first "
              "to include it in the comparison")

    basis, energy_ratio = experiment.fit_train_subspace(cfg, train)
    train_power = experiment.mean_power_spectrum(train)
    frame = experiment.evaluate_masks(
        mask_dict, test, cfg, run,
        prefix="manifold_compare", spectrum=train_power, subspace_basis=basis,
    )

    rows = []
    for name, mask in mask_dict.items():
        sub = frame[frame["mask"] == name]
        row = {
            "mask": name,
            "subspace_leakage": artifacts.subspace_nullspace_leakage(basis, mask),
            "mask_score": artifacts.expected_zero_filled_mse(mask, train_power),
        }
        for method in sorted(sub["method"].unique()):
            row[f"mse_{method}"] = float(sub[sub["method"] == method]["mse"].mean())
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(run / "metrics" / "manifold_comparison.csv", index=False)

    corr_rows = []
    for predictor in ("subspace_leakage", "mask_score"):
        for outcome in [c for c in table.columns if c.startswith("mse_")]:
            rho, p_value = spearmanr(table[predictor], table[outcome])
            corr_rows.append({"predictor": predictor, "outcome": outcome,
                              "spearman_rho": float(rho), "p_value": float(p_value)})
    corr = pd.DataFrame(corr_rows)
    corr.to_csv(run / "metrics" / "manifold_correlations.csv", index=False)

    viz.scatter_with_labels(
        table["subspace_leakage"].tolist(),
        table["mse_subspace"].tolist(),
        table["mask"].tolist(),
        run / "plots" / "manifold_leakage_vs_error.png",
        xlabel="subspace null-space leakage (design-time)",
        ylabel="mean subspace-recon MSE (test)",
        title="manifold coherence metric vs measured error",
    )

    print(f"\nsubspace d={basis.shape[1]} captures {energy_ratio:.4f} of train energy")
    print("\ncomparison table (manifold_comparison.csv):")
    print(table.round(5).to_string(index=False))
    print("\nrank agreement (does the design-time metric order the masks like the errors?):")
    print(corr.round(3).to_string(index=False))
    if "loupe_learned" in table["mask"].values:
        by_leak = table.sort_values("subspace_leakage")["mask"].tolist()
        by_err = table.sort_values("mse_subspace")["mask"].tolist()
        print(f"\nleakage ranking:  {' < '.join(by_leak)}")
        print(f"error ranking:    {' < '.join(by_err)}")
        print("(matching positions for loupe_learned mean the manifold metric "
              "predicts the learned mask's quality without training)")


if __name__ == "__main__":
    main()
