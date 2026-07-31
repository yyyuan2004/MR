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

from mrsim import artifacts, experiment, viz
from mrsim.config import load_config, run_dir, seed_everything
from mrsim.fft_ops import fft2c
from mrsim.progress import track

import torch


def active_support_atoms(
    train_images: np.ndarray, shape: tuple[int, int], wavelet: str, levels: int, support_size: int
) -> tuple[np.ndarray, list[str]]:
    """Frequency-domain columns F W*_S for the most active wavelet positions.

    Returns an (N_pixels, |S|) complex matrix whose columns are the centered
    spectra of the selected wavelet atoms, plus the subband label of every
    column ("approx" or "level<L>_<orientation>", level 1 = coarsest). The
    labels let sigma_min be resolved per subband: orientation encodes
    direction and the level hierarchy encodes scale.
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
    labels = []
    orientations = ("horizontal", "vertical", "diagonal")
    for _, band_index, detail_index, row, col in selected:
        blank = [np.zeros_like(template[0])] + [
            tuple(np.zeros_like(d) for d in b) for b in template[1:]
        ]
        target = blank[0] if band_index == 0 else blank[band_index][detail_index]
        target[row, col] = 1.0
        atom = pywt.waverec2(blank, wavelet=wavelet, mode="periodization")
        columns.append(fft2c(torch.from_numpy(atom)).numpy().ravel())
        labels.append(
            "approx" if band_index == 0 else f"level{band_index}_{orientations[detail_index]}"
        )
    return np.stack(columns, axis=1), labels


def atom_spectrum_pool(
    train_images: np.ndarray,
    shape: tuple[int, int],
    wavelet: str,
    levels: int,
    pool_size: int,
) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray]:
    """Cache the spectra of the most active wavelet positions, once.

    Drawing many supports is only affordable if the atom spectra are computed
    once instead of per support. The pool is the ``pool_size`` positions with
    the largest mean coefficient energy; supports are then column selections
    out of it. Also returns each position's energy (a sampling weight) and its
    parent index within the pool, or -1 when the parent is outside it, which is
    what the tree-structured sampler needs.
    """
    coeffs = pywt.wavedec2(
        train_images.astype(np.float64), wavelet=wavelet, mode="periodization",
        level=levels, axes=(-2, -1),
    )
    template = pywt.wavedec2(np.zeros(shape), wavelet=wavelet, mode="periodization", level=levels)

    entries = []  # (energy, band_index, detail_index, row, col)
    approx_energy = np.mean(np.abs(coeffs[0]) ** 2, axis=0)
    entries += [(approx_energy[r, c], 0, -1, r, c)
                for r in range(approx_energy.shape[0])
                for c in range(approx_energy.shape[1])]
    for band_index, band in enumerate(coeffs[1:], start=1):
        for detail_index, detail in enumerate(band):
            energy = np.mean(np.abs(detail) ** 2, axis=0)
            entries += [(energy[r, c], band_index, detail_index, r, c)
                        for r in range(energy.shape[0]) for c in range(energy.shape[1])]
    entries.sort(key=lambda e: -e[0])
    selected = entries[:pool_size]

    position_to_pool = {
        (band_index, detail_index, row, col): index
        for index, (_, band_index, detail_index, row, col) in enumerate(selected)
    }

    orientations = ("horizontal", "vertical", "diagonal")
    columns, labels, parents = [], [], []
    for _energy, band_index, detail_index, row, col in track(
        selected, total=len(selected), label="atom spectra"
    ):
        blank = [np.zeros_like(template[0])] + [
            tuple(np.zeros_like(d) for d in b) for b in template[1:]
        ]
        target = blank[0] if band_index == 0 else blank[band_index][detail_index]
        target[row, col] = 1.0
        atom = pywt.waverec2(blank, wavelet=wavelet, mode="periodization")
        columns.append(fft2c(torch.from_numpy(atom)).numpy().ravel().astype(np.complex64))
        labels.append(
            "approx" if band_index == 0 else f"level{band_index}_{orientations[detail_index]}"
        )
        # A detail coefficient's parent sits one level coarser at half the
        # spatial index; approximation coefficients have no parent.
        parents.append(
            position_to_pool.get((band_index - 1, detail_index, row // 2, col // 2), -1)
            if band_index > 1
            else -1
        )
    energies = np.array([entry[0] for entry in selected], dtype=np.float64)
    return np.stack(columns, axis=1), labels, energies, np.array(parents, dtype=np.int64)


def sample_supports(
    energies: np.ndarray,
    parents: np.ndarray,
    support_size: int,
    n_supports: int,
    rng: np.random.Generator,
    *,
    model: str = "energy",
    tree_boost: float = 8.0,
) -> list[np.ndarray]:
    """Draw candidate sparse supports from the pool.

    A single aggregated support turns a union-of-subspaces problem back into a
    single-subspace one, which is the linear-Gaussian regime where conditioning
    is largely determined by coverage. Sampling many supports restores the
    union structure, and the distribution of ``sigma_min`` over them -- not its
    value on one support -- is the quantity that can carry information coverage
    does not.

    ``model="energy"`` draws positions independently with probability
    proportional to mean coefficient energy. ``model="tree"`` additionally
    boosts a position whose parent is already in the support, because wavelet
    coefficients are not independent across scales: large coefficients persist
    along parent-child chains, so real supports are clustered in the tree
    rather than scattered.
    """
    if model not in {"energy", "tree"}:
        raise ValueError("model must be 'energy' or 'tree'")
    weights = np.maximum(energies, 1e-300)
    supports = []
    for _ in range(n_supports):
        if model == "energy":
            probability = weights / weights.sum()
            supports.append(
                rng.choice(weights.size, size=support_size, replace=False, p=probability)
            )
            continue
        chosen: list[int] = []
        available = np.ones(weights.size, dtype=bool)
        current = weights.copy()
        for _ in range(support_size):
            probability = np.where(available, current, 0.0)
            probability = probability / probability.sum()
            pick = int(rng.choice(weights.size, p=probability))
            chosen.append(pick)
            available[pick] = False
            # Children of the chosen position become more likely.
            current[parents == pick] *= tree_boost
        supports.append(np.array(chosen, dtype=np.int64))
    return supports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--support-size", type=int, default=256)
    parser.add_argument("--n-supports", type=int, default=24,
                        help="random supports drawn per sampling model (0 disables)")
    parser.add_argument("--pool-size", type=int, default=768,
                        help="most-active wavelet positions the supports are drawn from")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(int(cfg["seed"]))
    run = run_dir(cfg)
    images = experiment.load_or_generate_dataset(cfg, run)
    train, _, _ = experiment.train_val_test_split(images, cfg)

    shape, n_samples, _ = experiment.mask_budgets(cfg)
    ista_cfg = cfg.get("recon", {}).get("wavelet_ista", {})
    wavelet, levels = str(ista_cfg.get("wavelet", "db4")), int(ista_cfg.get("levels", 3))
    train_power = experiment.mean_power_spectrum(train)

    # sigma_min of an m x |S| matrix is structurally zero once |S| > m, and
    # near the boundary it measures conditioning of an almost-square system
    # rather than recoverability. Cap |S| at half the measurement budget so the
    # restricted system stays comfortably overdetermined for every mask.
    support_size = min(args.support_size, n_samples // 2)
    if support_size < args.support_size:
        print(f"capping support size {args.support_size} -> {support_size} "
              f"(measurement budget is {n_samples})")

    print(f"building the mask family and the |S|={support_size} active support")
    mask_dict = experiment.build_mask_family(cfg, train)
    atoms, atom_bands = active_support_atoms(train.numpy(), shape, wavelet, levels, support_size)
    band_order = list(dict.fromkeys(atom_bands))
    band_columns = {band: [i for i, b in enumerate(atom_bands) if b == band] for band in band_order}
    print(f"active support spans {len(band_order)} subbands: {', '.join(band_order)}")

    # wavelet_leakage is one of the repository's existing design-time
    # predictors, so it belongs in the axis comparison alongside the others.
    mass = artifacts.subband_spectral_mass(shape, wavelet=wavelet, levels=levels)
    energies = artifacts.subband_energies(
        train.detach().cpu().numpy(), wavelet=wavelet, levels=levels
    )

    rows = []
    spectra: dict[str, np.ndarray] = {}
    total_power = float(train_power.sum())
    for name in track(mask_dict, total=len(mask_dict), label="axis precheck"):
        mask = mask_dict[name]
        omega = np.flatnonzero(mask.ravel() > 0.5)
        restricted = atoms[omega]  # A F W*_S
        singular = np.linalg.svd(restricted, compute_uv=False)
        spectra[name] = singular
        # Standard numerical-rank tolerance; singular values below it are
        # unrecoverable directions, not measurements of conditioning.
        tol = float(singular.max()) * max(restricted.shape) * np.finfo(np.float64).eps
        n_deficient = int((singular <= tol).sum())
        # Per-subband sigma_min: orientation bands separate direction, the
        # level hierarchy separates scale, so this profile says *which*
        # directions/scales a mask leaves ill-conditioned, not just whether.
        band_sigma: dict[str, float] = {}
        for band in band_order:
            sub = restricted[:, band_columns[band]]
            band_singular = np.linalg.svd(sub, compute_uv=False)
            band_tol = float(band_singular.max()) * max(sub.shape) * np.finfo(np.float64).eps
            band_sigma[f"sigma_min_{band}"] = (
                float(band_singular.min()) if (band_singular <= band_tol).sum() == 0 else 0.0
            )
        rows.append({
            "mask": name,
            "rho": float((mask * train_power).sum() / total_power),
            "rho_real": artifacts.prior_observable_energy_fraction_real(mask, train_power),
            "effective_samples": artifacts.effective_sample_count(mask),
            "hermitian_redundancy": artifacts.hermitian_redundancy(mask),
            "psf_max_sidelobe": artifacts.psf_metrics(mask)["psf_max_sidelobe"],
            "wavelet_leakage": artifacts.wavelet_leakage_score(mask, mass, energies),
            "sigma_min": float(singular.min()) if n_deficient == 0 else 0.0,
            "n_deficient": n_deficient,
            "cond": float(singular.max() / max(singular.min(), 1e-12)),
            **band_sigma,
        })
    table = pd.DataFrame(rows)

    # Distribution of sigma_min over sampled supports. The fixed aggregated
    # support above collapses the union-of-subspaces structure; this restores it.
    if args.n_supports > 0:
        pool_atoms, _pool_labels, pool_energies, pool_parents = atom_spectrum_pool(
            train.numpy(), shape, wavelet, levels, args.pool_size
        )
        support_rng = np.random.default_rng(int(cfg["seed"]))
        support_sets = {
            model: sample_supports(
                pool_energies, pool_parents, support_size,
                args.n_supports, support_rng, model=model,
            )
            for model in ("energy", "tree")
        }
        distribution_rows = []
        for name in track(mask_dict, total=len(mask_dict), label="support ensemble"):
            omega = np.flatnonzero(mask_dict[name].ravel() > 0.5)
            restricted_all = pool_atoms[omega]
            for model, supports in support_sets.items():
                minima = []
                for support in supports:
                    block = restricted_all[:, support]
                    singular = np.linalg.svd(block, compute_uv=False)
                    tolerance = (
                        float(singular.max())
                        * max(block.shape)
                        * np.finfo(np.float32).eps
                    )
                    minima.append(
                        0.0 if (singular <= tolerance).any() else float(singular.min())
                    )
                minima = np.asarray(minima)
                distribution_rows.append({
                    "mask": name,
                    "support_model": model,
                    "n_supports": len(supports),
                    "sigma_min_median": float(np.median(minima)),
                    "sigma_min_p10": float(np.quantile(minima, 0.1)),
                    "sigma_min_p90": float(np.quantile(minima, 0.9)),
                    "p_rank_deficient": float(np.mean(minima == 0.0)),
                })
        distribution = pd.DataFrame(distribution_rows)
        distribution.to_csv(
            run / "metrics" / "axis_precheck_support_distribution.csv", index=False
        )
        print("\nsigma_min over sampled supports (restores the union-of-subspaces "
              "structure the single fixed support removes):")
        print(distribution.round(5).to_string(index=False))

        merged = distribution.merge(table[["mask", "rho"]], on="mask")
        print("\nrank correlation with coverage, per support model:")
        for model, block in merged.groupby("support_model"):
            usable = block[block["p_rank_deficient"] < 1.0]
            if usable["sigma_min_median"].nunique() < 3:
                print(f"  {model:8s}  skipped: too few distinct values")
                continue
            statistic = spearmanr(usable["rho"], usable["sigma_min_median"]).statistic
            verdict = (
                "REDUNDANT with coverage" if abs(statistic) > 0.8
                else "carries information coverage does not"
            )
            print(f"  {model:8s}  rho vs median sigma_min = {statistic:+.3f}   {verdict}")

    table.to_csv(run / "metrics" / "axis_precheck.csv", index=False)

    # Coverage and recoverability visualizations on a readable subset.
    subset = viz.representative_subset(mask_dict, train_power, n=8)
    viz.plot_radial_coverage(subset, train_power, run / "plots" / "radial_coverage.png")
    viz.plot_subband_leakage(subset, mass, energies, run / "plots" / "subband_leakage.png")
    viz.plot_singular_spectra(
        {name: spectra[name] for name in subset}, run / "plots" / "singular_spectra.png"
    )
    profiles = {
        name: {band: float(table.loc[table["mask"] == name, f"sigma_min_{band}"].iloc[0])
               for band in band_order}
        for name in subset
    }
    viz.plot_subband_sigma_min(profiles, run / "plots" / "subband_sigma_min.png")
    print("\nper-subband sigma_min (representative subset; 0 = band has "
          "unrecoverable directions):")
    profile_table = pd.DataFrame(profiles).T
    print(profile_table.round(4).to_string())

    print(f"\n{len(table)} masks\n")
    print(table.sort_values("rho").round(4).to_string(index=False))

    print("\nrank correlations between candidate axes:")
    pairs = [
        ("rho", "psf_max_sidelobe"),
        ("rho", "sigma_min"),
        ("rho", "wavelet_leakage"),
        ("psf_max_sidelobe", "sigma_min"),
        ("psf_max_sidelobe", "wavelet_leakage"),
        ("sigma_min", "wavelet_leakage"),
    ]
    # Beyond rank statistics: does sigma_min still vary once coverage is held
    # fixed? Bin the full-rank masks by rho and report the within-bin spread.
    # Large within-bin dispersion means conditioning carries information that
    # coverage does not; near-zero dispersion means the axis is redundant.
    full_rank_rows = table[table["n_deficient"] == 0].copy()
    if len(full_rank_rows) >= 6:
        n_bins_rho = min(4, len(full_rank_rows) // 3)
        full_rank_rows["rho_bin"] = pd.qcut(full_rank_rows["rho"], q=n_bins_rho, duplicates="drop")
        log_sigma = np.log10(full_rank_rows["sigma_min"])
        dispersion = (
            full_rank_rows.assign(log10_sigma_min=log_sigma)
            .groupby("rho_bin", observed=True)
            .agg(
                n=("mask", "size"),
                rho_lo=("rho", "min"),
                rho_hi=("rho", "max"),
                sigma_min_lo=("sigma_min", "min"),
                sigma_min_hi=("sigma_min", "max"),
                log10_spread=("log10_sigma_min", lambda s: float(s.max() - s.min())),
                log10_std=("log10_sigma_min", "std"),
            )
            .reset_index(drop=True)
        )
        dispersion.to_csv(run / "metrics" / "axis_precheck_dispersion.csv", index=False)
        # Variance split of log10 sigma_min: how much is left at fixed coverage?
        grouped = full_rank_rows.assign(v=log_sigma).groupby("rho_bin", observed=True)["v"]
        within = float((grouped.transform("mean") - log_sigma).pow(2).mean())
        total = float(log_sigma.var(ddof=0))
        print("\nsigma_min dispersion within fixed-rho bins (full-rank masks):")
        print(dispersion.round(4).to_string(index=False))
        print(f"within-bin share of log10(sigma_min) variance: {within / max(total, 1e-30):.2f}"
              " (near 0 = redundant with coverage; near 1 = independent information)")

        viz.scatter_with_labels(
            full_rank_rows["rho"].tolist(),
            full_rank_rows["sigma_min"].tolist(),
            full_rank_rows["mask"].tolist(),
            run / "plots" / "rho_vs_sigma_min.png",
            xlabel="rho (observed training-energy fraction)",
            ylabel="sigma_min of the restricted operator",
            title="coverage vs restricted conditioning (full-rank masks)",
        )

    # Rank-deficient masks have sigma_min = 0 exactly; ranking them against
    # each other is numerical noise, so sigma_min pairs use only full-rank rows.
    full_rank = table[table["n_deficient"] == 0]
    n_dropped = len(table) - len(full_rank)
    if n_dropped:
        print(f"  (sigma_min pairs computed on {len(full_rank)} full-rank masks; "
              f"{n_dropped} rank-deficient masks excluded)")
    for a, b in pairs:
        source = full_rank if "sigma_min" in (a, b) else table
        if len(source) < 3:
            print(f"  {a:18s} vs {b:18s}  skipped: too few full-rank masks")
            continue
        rho, p = spearmanr(source[a], source[b])
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
