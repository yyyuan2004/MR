#!/usr/bin/env python3
"""B4 pre-check: are the proposed sweet-spot axes independent?

Before building a 2-D diagnostic, measure whether its two axes carry different
information. Computes, over the whole mask family:

  rho(A)            observed fraction of training spectral energy
  psf_max_sidelobe  global PSF coherence
  sigma_min         smallest singular value of A F W*_S, the restricted
                    conditioning of the measurement operator on the empirically
                    active wavelet support S

and reports their rank correlations. If a candidate y-axis correlates with
rho above ~0.8 the plane collapses to a Pareto band and the axis is redundant.

Definition of the active support S (fixed here, and reproducible): aggregate
the mean squared wavelet coefficient over ALL training images, then take the
`support_size` largest coefficient positions. It is aggregated, not per-image,
so one support serves every mask and the comparison is like-for-like.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pywt
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mrsim import artifacts, experiment
from mrsim.config import load_config, run_dir, seed_everything
from mrsim.fft_ops import fft2c
from mrsim.progress import track

import torch


def active_support_atoms(
    train_images: np.ndarray, shape: tuple[int, int], wavelet: str, levels: int, support_size: int
) -> np.ndarray:
    """Frequency-domain columns F W*_S for the most active wavelet positions.

    Returns an (N_pixels, |S|) complex matrix whose columns are the centered
    spectra of the selected wavelet atoms.
    """
    coeffs = pywt.wavedec2(
        train_images.astype(np.float64), wavelet=wavelet, mode="periodization",
        level=levels, axes=(-2, -1),
    )
    template = pywt.wavedec2(np.zeros(shape), wavelet=wavelet, mode="periodization", level=levels)

    # Flatten every subband's mean energy into one ranking over all positions.
    entries = []  # (energy, band_index, detail_index, row, col)
    mean_energy = [np.mean(np.abs(coeffs[0]) ** 2, axis=0)]
    entries += [(mean_energy[0][r, c], 0, -1, r, c)
                for r in range(mean_energy[0].shape[0])
                for c in range(mean_energy[0].shape[1])]
    for band_index, band in enumerate(coeffs[1:], start=1):
        for detail_index, detail in enumerate(band):
            energy = np.mean(np.abs(detail) ** 2, axis=0)
            entries += [(energy[r, c], band_index, detail_index, r, c)
                        for r in range(energy.shape[0]) for c in range(energy.shape[1])]
    entries.sort(key=lambda e: -e[0])
    selected = entries[:support_size]

    columns = []
    for _, band_index, detail_index, row, col in selected:
        blank = [np.zeros_like(template[0])] + [
            tuple(np.zeros_like(d) for d in b) for b in template[1:]
        ]
        target = blank[0] if band_index == 0 else blank[band_index][detail_index]
        target[row, col] = 1.0
        atom = pywt.waverec2(blank, wavelet=wavelet, mode="periodization")
        columns.append(fft2c(torch.from_numpy(atom)).numpy().ravel())
    return np.stack(columns, axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--support-size", type=int, default=256)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(int(cfg["seed"]))
    run = run_dir(cfg)
    images = experiment.load_or_generate_dataset(cfg, run)
    train, _, _ = experiment.train_val_test_split(images, cfg)

    shape, _, _ = experiment.mask_budgets(cfg)
    ista_cfg = cfg.get("recon", {}).get("wavelet_ista", {})
    wavelet, levels = str(ista_cfg.get("wavelet", "db4")), int(ista_cfg.get("levels", 3))
    train_power = experiment.mean_power_spectrum(train)

    print(f"building the mask family and the |S|={args.support_size} active support")
    mask_dict = experiment.build_mask_family(cfg, train)
    atoms = active_support_atoms(train.numpy(), shape, wavelet, levels, args.support_size)

    rows = []
    total_power = float(train_power.sum())
    for name in track(mask_dict, total=len(mask_dict), label="axis precheck"):
        mask = mask_dict[name]
        omega = np.flatnonzero(mask.ravel() > 0.5)
        restricted = atoms[omega]  # A F W*_S
        singular = np.linalg.svd(restricted, compute_uv=False)
        rows.append({
            "mask": name,
            "rho": float((mask * train_power).sum() / total_power),
            "psf_max_sidelobe": artifacts.psf_metrics(mask)["psf_max_sidelobe"],
            "wavelet_leakage": None,
            "sigma_min": float(singular.min()),
            "cond": float(singular.max() / max(singular.min(), 1e-12)),
        })
    table = pd.DataFrame(rows).drop(columns=["wavelet_leakage"])
    table.to_csv(run / "metrics" / "axis_precheck.csv", index=False)

    print(f"\n{len(table)} masks\n")
    print(table.sort_values("rho").round(4).to_string(index=False))

    print("\nrank correlations between candidate axes:")
    pairs = [("rho", "psf_max_sidelobe"), ("rho", "sigma_min"), ("psf_max_sidelobe", "sigma_min")]
    for a, b in pairs:
        rho, p = spearmanr(table[a], table[b])
        verdict = "REDUNDANT (|rho| > 0.8)" if abs(rho) > 0.8 else "independent enough"
        print(f"  {a:18s} vs {b:18s}  rho={rho:+.3f}  p={p:.4f}   {verdict}")

    print(f"\nrho dynamic range:       [{table['rho'].min():.4f}, {table['rho'].max():.4f}]")
    print(f"sidelobe dynamic range:  [{table['psf_max_sidelobe'].min():.4f}, "
          f"{table['psf_max_sidelobe'].max():.4f}]")
    print(f"sigma_min dynamic range: [{table['sigma_min'].min():.4g}, "
          f"{table['sigma_min'].max():.4g}]")
    print(f"\ntable: {run / 'metrics' / 'axis_precheck.csv'}")


if __name__ == "__main__":
    main()
