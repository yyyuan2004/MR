#!/usr/bin/env python3
"""Compare full-Cartesian-line designs against the learned-mask baseline:
equispaced and variable-density baselines, diagonal-prior line A-optimal,
wavelet-subspace leakage, and the learned probabilistic line mask from script
09 (if it has been trained).

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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mrsim import artifacts, experiment, viz
from mrsim.config import load_config, run_dir, save_config_snapshot, seed_everything

# The learned sampler in script 09 selects complete Cartesian columns.  Keep
# this comparison within the same acquisition family; point masks have a
# different hardware constraint and are reported by script 07 instead.
MODEL_BASED = [
    "equispaced_lines",
    "variable_density_lines",
    "line_aopt",
    "line_subspace_leakage",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    rng = seed_everything(int(cfg["seed"]))
    run = run_dir(cfg)
    save_config_snapshot(cfg, run, "10_compare_manifold_vs_learned", cli_args=vars(args))

    images = experiment.load_or_generate_dataset(cfg, run)
    train, _, test = experiment.train_validation_test_split(images, cfg)

    print(f"building masks: {', '.join(MODEL_BASED)}")
    mask_dict = experiment.build_masks(MODEL_BASED, cfg, train, rng)
    loupe_path = run / "masks" / "loupe_learned.npy"
    if loupe_path.exists():
        mask_dict["loupe_learned"] = np.load(loupe_path)
    else:
        print(f"no learned mask at {loupe_path}; run scripts/09_loupe_baseline.py first "
              "to include it in the comparison")

    statistics = experiment.fit_train_subspace_statistics(cfg, train)
    basis = statistics.basis
    energy_ratio = statistics.energy_ratio
    prior_mean, prior_variance, train_power = experiment.frequency_prior_statistics(train)
    frame = experiment.evaluate_masks(
        mask_dict, test, cfg, run,
        prefix="manifold_compare",
        spectrum=prior_variance,
        prior_mean=prior_mean,
        subspace_basis=basis,
        subspace_mean=statistics.mean_image,
        subspace_variances=statistics.eigenvalues,
    )

    rows = []
    for name, mask in mask_dict.items():
        sub = frame[frame["mask"] == name]
        row = {
            "mask": name,
            "actual_n_samples": int(mask.sum()),
            "actual_n_lines": int((mask.sum(axis=0) > 0).sum()),
            "actual_acceleration": float(mask.size / mask.sum()),
            "acquisition_family": experiment.acquisition_family(mask),
            "subspace_leakage": artifacts.subspace_nullspace_leakage(
                basis,
                mask,
                weights=np.sqrt(statistics.eigenvalues),
            ),
            "mask_score": artifacts.expected_zero_filled_mse(
                mask,
                train_power,
                noise_std=float(cfg.get("measurement", {}).get("noise_std", 0.0)),
            ),
        }
        for method in sorted(sub["method"].unique()):
            block = sub[sub["method"] == method]
            row[f"complex_mse_{method}"] = float(block["complex_mse"].mean())
            row[f"magnitude_mse_{method}"] = float(block["magnitude_mse"].mean())
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(run / "metrics" / "manifold_comparison.csv", index=False)

    outcomes = [c for c in table.columns if c.startswith("complex_mse_")]
    corr = experiment.rank_correlations(
        table,
        predictors=["subspace_leakage", "mask_score"],
        outcomes=outcomes,
    )
    corr.to_csv(run / "metrics" / "manifold_correlations.csv", index=False)

    viz.scatter_with_labels(
        table["subspace_leakage"].tolist(),
        table["complex_mse_subspace"].tolist(),
        table["mask"].tolist(),
        run / "plots" / "manifold_leakage_vs_error.png",
        xlabel="subspace null-space leakage (design-time)",
        ylabel="mean subspace-recon complex MSE (test)",
        title="manifold coherence metric vs measured error",
    )

    print(
        f"\nsubspace d={basis.shape[1]} explains "
        f"{energy_ratio:.4f} of centered train variance"
    )
    print("\ncomparison table (manifold_comparison.csv):")
    print(table.round(5).to_string(index=False))
    print("\nrank agreement (does the design-time metric order the masks like the errors?):")
    print(corr.round(3).to_string(index=False))
    if "loupe_learned" in table["mask"].values:
        by_leak = table.sort_values("subspace_leakage")["mask"].tolist()
        by_err = table.sort_values("complex_mse_subspace")["mask"].tolist()
        print(f"\nleakage ranking:  {' < '.join(by_leak)}")
        print(f"error ranking:    {' < '.join(by_err)}")
        print(
            "These rankings are descriptive diagnostics over a small, "
            "constructed mask set; matching positions are not confirmatory evidence."
        )


if __name__ == "__main__":
    main()
