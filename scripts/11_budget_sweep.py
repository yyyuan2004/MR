#!/usr/bin/env python3
"""Sweep the measurement budget across acceleration factors.

At 4x the whole low-frequency block fits inside the budget, so sampling
design barely matters — every sensible mask takes the same energy-dense core.
The capture-energy / control-coherence trade-off only bites once the budget is
too small for that core, which is what 8x and 12x test. Reusing
build_mask_family and evaluate_masks, this concatenates the per-budget
argumentation tables into one long table (metrics/budget_sweep.csv) and prints
three decisive numbers per budget:

  1. Does the PSF penalty still collapse onto plain A-optimal? (Jaccard)
  2. Does the spectral energy score still predict the nonlinear error? (rho)
  3. Does a null-space norm predict whether the nonlinear prior pays off?

Outputs also feed scripts/13_phase_diagram.py.
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mrsim import artifacts, experiment, masks
from mrsim.config import load_config, run_dir, save_config_snapshot, seed_everything

PREDICTORS = [
    "mask_score",
    "wavelet_leakage",
    "subspace_leakage",
    "weighted_max_sidelobe",
    "psf_max_sidelobe",
    "truth_nullspace_norm",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--n-test", type=int, default=None, help="subsample the test split")
    parser.add_argument(
        "--examples", action="store_true",
        help="also render per-mask example grids (slow: 3 figures per mask/method)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(int(cfg["seed"]))
    run = run_dir(cfg)
    save_config_snapshot(cfg, run, "11_budget_sweep")

    images = experiment.load_or_generate_dataset(cfg, run)
    train, test = experiment.train_test_split(images, cfg)
    if args.n_test:
        test = test[: args.n_test]

    train_power = experiment.mean_power_spectrum(train)
    basis, _ = experiment.fit_train_subspace(cfg, train)
    ista_cfg = cfg.get("recon", {}).get("wavelet_ista", {})
    wavelet, levels = str(ista_cfg.get("wavelet", "db4")), int(ista_cfg.get("levels", 3))
    mass = artifacts.subband_spectral_mass(train_power.shape, wavelet=wavelet, levels=levels)
    energies = artifacts.subband_energies(train.numpy(), wavelet=wavelet, levels=levels)
    unet = experiment.load_unet(run)
    if unet is None:
        print("no trained U-Net found; run scripts/12_train_unet.py to add the learned arm")

    accelerations = [float(a) for a in cfg.get("budget_sweep", {}).get("accelerations", [4, 8, 12])]
    tables, correlations = [], []
    for acceleration in accelerations:
        budget_cfg = copy.deepcopy(cfg)
        budget_cfg["mask"]["sampling_fraction"] = 1.0 / acceleration
        shape, n_samples, _ = experiment.mask_budgets(budget_cfg)
        sub_run = run / f"budget_{acceleration:g}x"
        print(f"\n=== acceleration {acceleration:g}x  ({n_samples} of {shape[0] * shape[1]}) ===")

        mask_dict = experiment.build_mask_family(budget_cfg, train)
        frame = experiment.evaluate_masks(
            mask_dict, test, budget_cfg, sub_run, prefix=f"budget_{acceleration:g}x",
            spectrum=train_power, subspace_basis=basis, unet_model=unet,
            write_examples=args.examples,
        )
        table = experiment.argumentation_table(
            mask_dict, frame, train_power, mass=mass, energies=energies, basis=basis
        )
        table.insert(0, "acceleration", acceleration)
        table.insert(1, "n_samples", n_samples)

        # Focus 1: has the PSF penalty separated from plain A-optimal yet?
        aopt = mask_dict["aopt_greedy"]
        table["jaccard_vs_aopt"] = [masks.jaccard(mask_dict[m], aopt) for m in table["mask"]]

        outcomes = [c for c in table.columns if c.startswith(("mse_", "psnr_gain_"))]
        corr = experiment.rank_correlations(table, PREDICTORS, outcomes)
        corr.insert(0, "acceleration", acceleration)
        tables.append(table)
        correlations.append(corr)

    long_table = pd.concat(tables, ignore_index=True)
    long_corr = pd.concat(correlations, ignore_index=True)
    long_table.to_csv(run / "metrics" / "budget_sweep.csv", index=False)
    long_corr.to_csv(run / "metrics" / "budget_sweep_correlations.csv", index=False)

    print(f"\n{'=' * 78}\nBUDGET SWEEP: {len(long_table)} mask-budget rows "
          f"({len(tables[0])} masks x {len(accelerations)} budgets)\n{'=' * 78}")

    print("\n[1] PSF-penalized vs plain A-optimal (Jaccard; 1.0 = identical mask)")
    penalized = long_table[long_table["mask"].str.startswith("psf_penalized")]
    print(
        penalized.pivot_table(index="mask", columns="acceleration", values="jaccard_vs_aopt")
        .round(4).to_string()
    )

    print("\n[2] Does the spectral energy score still predict error? (Spearman rho)")
    energy_rows = long_corr[
        (long_corr["predictor"] == "mask_score")
        & (long_corr["outcome"].isin(["mse_zero_filled", "mse_wavelet_ista", "mse_unet_post"]))
    ]
    if len(energy_rows):
        print(
            energy_rows.pivot_table(index="outcome", columns="acceleration", values="spearman_rho")
            .round(3).to_string()
        )

    print("\n[3] Do null-space norms predict the nonlinear prior's payoff?")
    gain_cols = [c for c in long_table.columns if c.startswith("psnr_gain_")]
    gain_rows = long_corr[long_corr["outcome"].isin(gain_cols)]
    if len(gain_rows):
        pivot = gain_rows.pivot_table(
            index=["predictor", "outcome"], columns="acceleration", values="spearman_rho"
        )
        print(pivot.round(3).to_string())
        best = gain_rows.loc[gain_rows["p_value"].idxmin()]
        print(
            f"\nstrongest gain predictor: {best['predictor']} -> {best['outcome']} "
            f"at {best['acceleration']:g}x  (rho={best['spearman_rho']:.3f}, "
            f"p={best['p_value']:.4f}, n={int(best['n_masks'])})"
        )
        for col in gain_cols:
            per_budget = long_table.groupby("acceleration")[col].mean()
            print(f"mean {col} by acceleration: "
                  + ", ".join(f"{a:g}x={v:+.2f}dB" for a, v in per_budget.items()))

    print(f"\nlong table:   {run / 'metrics' / 'budget_sweep.csv'}")
    print(f"correlations: {run / 'metrics' / 'budget_sweep_correlations.csv'}")


if __name__ == "__main__":
    main()
