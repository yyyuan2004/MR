#!/usr/bin/env python3
"""Sweep the measurement budget across acceleration factors.

This exploratory script applies the acceleration factors declared by
``budget_sweep.accelerations``. Reusing ``build_mask_family`` and
``evaluate_masks``, it concatenates per-budget argumentation tables into one
long table and prints three descriptive diagnostics per budget:

  1. Does the PSF penalty still collapse onto plain A-optimal? (Jaccard)
  2. Does the spectral energy score still predict the nonlinear error? (rho)
  3. Does a null-space norm predict whether the nonlinear prior pays off?

Rank correlations are descriptive diagnostics over a deliberately constructed
mask family; they are not treated as independent-sample significance tests.
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
    "prior_unobservable_energy_fraction",
    "wavelet_leakage",
    "subspace_leakage",
    "weighted_max_sidelobe",
    "psf_max_sidelobe",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--n-test", type=int, default=None, help="subsample the test split")
    parser.add_argument(
        "--examples", action="store_true",
        help="also render per-mask example grids (slow: 3 figures per mask/method)",
    )
    parser.add_argument(
        "--tune-ista", action="store_true",
        help="select the ISTA threshold per mask on the validation split",
    )
    parser.add_argument(
        "--threshold-grid", type=float, nargs="+", default=None,
        help="override recon.wavelet_ista.threshold_grid; tuning costs one extra "
             "ISTA per mask per grid point, so trim the grid for large runs",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(int(cfg["seed"]))
    run = run_dir(cfg)
    save_config_snapshot(cfg, run, "11_budget_sweep", cli_args=vars(args))

    images = experiment.load_or_generate_dataset(cfg, run)
    train, validation, test = experiment.train_validation_test_split(images, cfg)
    if args.n_test:
        test = test[: args.n_test]
    if args.threshold_grid:
        cfg["recon"]["wavelet_ista"]["threshold_grid"] = list(args.threshold_grid)
    if args.tune_ista and validation.shape[0] == 0:
        raise ValueError("--tune-ista requires data.n_val > 0")
    if args.tune_ista:
        grid = cfg["recon"]["wavelet_ista"].get(
            "threshold_grid", experiment.DEFAULT_THRESHOLD_GRID
        )
        print(f"tuning the ISTA threshold per mask on {validation.shape[0]} validation "
              f"images over {len(grid)} grid points")

    prior_mean, prior_variance, train_power = experiment.frequency_prior_statistics(train)
    subspace_statistics = experiment.fit_train_subspace_statistics(cfg, train)
    basis = subspace_statistics.basis
    ista_cfg = cfg.get("recon", {}).get("wavelet_ista", {})
    wavelet, levels = str(ista_cfg.get("wavelet", "db4")), int(ista_cfg.get("levels", 3))
    mass = artifacts.subband_spectral_mass(train_power.shape, wavelet=wavelet, levels=levels)
    energies = artifacts.subband_energies(
        train.detach().cpu().numpy(), wavelet=wavelet, levels=levels
    )
    include_unet = bool(
        cfg.get("experimental", {}).get("include_unet_in_comparisons", False)
    )
    unet = experiment.load_unet(run, cfg=cfg) if include_unet else None
    if include_unet and unet is None:
        print("no compatible trained U-Net found; run scripts/12_train_unet.py first")

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
            spectrum=prior_variance,
            prior_mean=prior_mean,
            subspace_basis=basis,
            subspace_mean=subspace_statistics.mean_image,
            subspace_variances=subspace_statistics.eigenvalues,
            unet_model=unet,
            write_examples=args.examples,
            # Tuning runs per mask inside evaluate_masks, so it covers the whole
            # mask family, not just the named masks script 07 builds.
            val_images=validation if args.tune_ista else None,
        )
        table = experiment.argumentation_table(
            mask_dict,
            frame,
            train_power,
            mass=mass,
            energies=energies,
            basis=subspace_statistics.covariance_factor,
            noise_std=float(
                budget_cfg.get("measurement", {}).get("noise_std", 0.0)
            ),
        )
        table.insert(0, "requested_acceleration", acceleration)
        table.insert(1, "requested_point_budget", n_samples)

        # Focus 1: has the PSF penalty separated from plain A-optimal yet?
        aopt = mask_dict["aopt_greedy"]
        table["jaccard_vs_aopt"] = [masks.jaccard(mask_dict[m], aopt) for m in table["mask"]]

        outcomes = [
            c
            for c in table.columns
            if c.startswith(("complex_mse_", "magnitude_mse_", "psnr_gain_"))
        ]
        tables.append(table)
        for acquisition_family, family_table in table.groupby("acquisition_family"):
            corr = experiment.rank_correlations(
                family_table, PREDICTORS, outcomes
            )
            corr.insert(0, "acquisition_family", acquisition_family)
            corr.insert(0, "requested_acceleration", acceleration)
            correlations.append(corr)

    long_table = pd.concat(tables, ignore_index=True)
    long_corr = pd.concat(correlations, ignore_index=True)
    (run / "metrics").mkdir(parents=True, exist_ok=True)
    long_table.to_csv(run / "metrics" / "budget_sweep.csv", index=False)
    long_corr.to_csv(run / "metrics" / "budget_sweep_correlations.csv", index=False)

    print(f"\n{'=' * 78}\nBUDGET SWEEP: {len(long_table)} mask-budget rows "
          f"({len(tables[0])} masks x {len(accelerations)} budgets)\n{'=' * 78}")

    print("\n[1] PSF-penalized vs plain A-optimal (Jaccard; 1.0 = identical mask)")
    penalized = long_table[long_table["mask"].str.startswith("psf_penalized")]
    print(
        penalized.pivot_table(
            index="mask",
            columns="requested_acceleration",
            values="jaccard_vs_aopt",
        )
        .round(4).to_string()
    )

    print("\n[2] Does the spectral energy score still predict error? (Spearman rho)")
    energy_rows = long_corr[
        (long_corr["predictor"] == "mask_score")
        & (
            long_corr["outcome"].isin(
                [
                    "complex_mse_zero_filled",
                    "complex_mse_wavelet_ista",
                    "complex_mse_unet_post",
                ]
            )
        )
    ]
    if len(energy_rows):
        print(
            energy_rows.pivot_table(
                index=["acquisition_family", "outcome"],
                columns="requested_acceleration",
                values="spearman_rho",
            )
            .round(3).to_string()
        )

    print("\n[3] Do null-space norms predict the nonlinear prior's payoff?")
    gain_cols = [c for c in long_table.columns if c.startswith("psnr_gain_")]
    gain_rows = long_corr[long_corr["outcome"].isin(gain_cols)]
    if len(gain_rows):
        pivot = gain_rows.pivot_table(
            index=["acquisition_family", "predictor", "outcome"],
            columns="requested_acceleration",
            values="spearman_rho",
        )
        print(pivot.round(3).to_string())
        best = gain_rows.loc[gain_rows["spearman_rho"].abs().idxmax()]
        print(
            f"\nstrongest descriptive gain association: "
            f"{best['predictor']} -> {best['outcome']} "
            f"at requested {best['requested_acceleration']:g}x  "
            f"(rho={best['spearman_rho']:.3f}, n={int(best['n_masks'])})"
        )
        for col in gain_cols:
            per_budget = long_table.groupby("requested_acceleration")[col].mean()
            print(f"mean {col} by acceleration: "
                  + ", ".join(f"{a:g}x={v:+.2f}dB" for a, v in per_budget.items()))

    print(f"\nlong table:   {run / 'metrics' / 'budget_sweep.csv'}")
    print(f"correlations: {run / 'metrics' / 'budget_sweep_correlations.csv'}")


if __name__ == "__main__":
    main()
