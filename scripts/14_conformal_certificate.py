#!/usr/bin/env python3
"""Split-conformal error certificates for each mask and reconstruction method.

The repository reports oracle error decompositions, which need the ground truth
and so cannot certify anything about an unseen sample. This script produces the
statement that *is* available at test time: calibrate the reconstruction error
on the validation split, and the resulting threshold covers a fresh exchangeable
sample with probability at least ``1 - alpha``.

Read the output with its two caveats in view. The guarantee is *marginal*, over
the joint draw of calibration and test points -- it says nothing about any
individual image, and distribution-free *conditional* coverage is known to be
unattainable. And it assumes exchangeability between the calibration and test
splits, which holds here only because both come from the same generator; under
the prior shift this package contemplates elsewhere, it does not.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mrsim import certify, experiment
from mrsim.config import load_config, run_dir, save_config_snapshot, seed_everything
from mrsim.progress import track


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--alpha", type=float, default=0.1)
    args = parser.parse_args()

    cfg = load_config(args.config)
    rng = seed_everything(int(cfg["seed"]))
    run = run_dir(cfg)
    save_config_snapshot(cfg, run, "14_conformal_certificate", cli_args=vars(args))

    images = experiment.load_or_generate_dataset(cfg, run)
    train, validation, test = experiment.train_validation_test_split(images, cfg)
    if validation.shape[0] == 0:
        raise SystemExit("conformal calibration needs a non-empty validation split")

    mask_dict = experiment.build_masks(list(cfg["mask"]["types"]), cfg, train, rng)
    prior_mean, prior_variance, _ = experiment.frequency_prior_statistics(train)

    rows = []
    for name in track(mask_dict, total=len(mask_dict), label="conformal"):
        mask = mask_dict[name]
        calibration = experiment.reconstruct_all(
            validation, mask, cfg, spectrum=prior_variance, prior_mean=prior_mean
        )
        evaluation = experiment.reconstruct_all(
            test, mask, cfg, spectrum=prior_variance, prior_mean=prior_mean
        )
        for method in sorted(calibration):
            if method not in evaluation:
                continue
            calibration_scores = certify.error_scores(
                calibration[method].detach().cpu().numpy(), validation.numpy()
            )
            test_scores = certify.error_scores(
                evaluation[method].detach().cpu().numpy(), test.numpy()
            )
            certificate = certify.calibrate(calibration_scores, args.alpha)
            low, high = certify.coverage_interval(test.shape[0], args.alpha)
            rows.append(
                {
                    "mask": name,
                    "method": method,
                    "alpha": args.alpha,
                    "n_calibration": certificate.n_calibration,
                    "n_test": int(test.shape[0]),
                    "threshold": certificate.threshold,
                    "nominal_coverage": certificate.nominal_coverage,
                    "empirical_coverage": certificate.empirical_coverage(test_scores),
                    "coverage_interval_low": low,
                    "coverage_interval_high": high,
                    "median_test_error": float(np.median(test_scores)),
                }
            )

    table = pd.DataFrame(rows)
    destination = run / "metrics" / "conformal_certificates.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(destination, index=False)

    print(table.round(4).to_string(index=False))
    inside = table[
        (table["empirical_coverage"] >= table["coverage_interval_low"])
        & (table["empirical_coverage"] <= table["coverage_interval_high"])
    ]
    print(
        f"\n{len(inside)}/{len(table)} (mask, method) pairs land inside the "
        f"{int(100 * (1 - args.alpha))}% nominal interval"
    )
    if int(test.shape[0]) < 100:
        print(
            f"note: with {int(test.shape[0])} test images the coverage interval is "
            f"[{table['coverage_interval_low'].iloc[0]:.3f}, "
            f"{table['coverage_interval_high'].iloc[0]:.3f}] -- wide enough that "
            "almost nothing is distinguishable. Raise data.n_test before reading "
            "any of these numbers as evidence."
        )
    print(f"\ntable: {destination}")


if __name__ == "__main__":
    main()
