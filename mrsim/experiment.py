"""Shared experiment plumbing used by the numbered scripts."""

from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch

from . import artifacts, data, greedy, masks, metrics, recon, subspace, viz
from .fft_ops import fft2c
from .progress import track


# ---------------------------------------------------------------------------
# Dataset handling
# ---------------------------------------------------------------------------

def dataset_path(run: Path) -> Path:
    return run / "data" / "dataset.pt"


def load_or_generate_dataset(cfg: dict[str, Any], run: Path, force: bool = False) -> torch.Tensor:
    """Load the run's dataset, generating and saving it if needed."""
    path = dataset_path(run)
    if path.exists() and not force:
        return torch.load(path)
    d = cfg["data"]
    images = data.generate_dataset(
        n_images=int(d["n_images"]),
        size=int(d["image_size"]),
        seed=int(cfg["seed"]),
        phantom=str(d.get("phantom", "ellipses")),
        min_ellipses=int(d.get("min_ellipses", 3)),
        max_ellipses=int(d.get("max_ellipses", 8)),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(images, path)
    return images


def train_test_split(images: torch.Tensor, cfg: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic split: first n_train images train, next n_test images test."""
    d = cfg["data"]
    n_train, n_test = int(d["n_train"]), int(d["n_test"])
    if n_train + n_test > images.shape[0]:
        raise ValueError("n_train + n_test exceeds the dataset size")
    return images[:n_train], images[n_train : n_train + n_test]


# ---------------------------------------------------------------------------
# Spectra and mask construction
# ---------------------------------------------------------------------------

def mean_power_spectrum(images: torch.Tensor) -> np.ndarray:
    """Mean centered frequency-domain power |X_k|^2 over a stack of images."""
    k = fft2c(images.to(torch.complex64))
    return (k.abs() ** 2).mean(dim=0).numpy().astype(np.float64)


def fitted_power_law_spectrum(images: torch.Tensor) -> np.ndarray:
    """Radially symmetric power-law fit to the empirical mean spectrum.

    Fits log s = intercept + slope * log(1 + r) by least squares and rebuilds
    a smooth spectrum from the radius map. Used as the model prior for
    A-optimal and artifact-aware greedy selection.
    """
    empirical = mean_power_spectrum(images)
    r = masks.radius_map(empirical.shape)
    x = np.log1p(r).ravel()
    y = np.log(empirical.ravel() + 1e-12)
    slope, intercept = np.polyfit(x, y, 1)
    spectrum = np.exp(intercept) * (1.0 + r) ** slope
    return np.maximum(spectrum, 1e-12)


def load_unet(run: Path):
    """Load the trained U-Net post-processor if scripts/12 has produced one."""
    path = run / "models" / "unet_post.pt"
    if not path.exists():
        return None
    from .unet import SmallUNet

    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = SmallUNet(base_channels=int(payload.get("base_channels", 16)))
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def fit_train_subspace(
    cfg: dict[str, Any], train_images: torch.Tensor
) -> tuple[np.ndarray, float]:
    """Fit the config-sized subspace basis on the train split."""
    d = int(cfg.get("subspace", {}).get("d", 32))
    return subspace.fit_subspace(train_images, d)


def greedy_noise_var(cfg: dict[str, Any]) -> float:
    """Noise variance assumed by the greedy selection criteria.

    Invariant: defaults to measurement.noise_std ** 2 so the criterion assumes
    exactly the noise that is simulated. greedy.noise_var overrides only when
    set explicitly in the config.
    """
    g = cfg.get("greedy", {})
    if "noise_var" in g and g["noise_var"] is not None:
        return float(g["noise_var"])
    return float(cfg.get("measurement", {}).get("noise_std", 0.0)) ** 2


def mask_budgets(cfg: dict[str, Any]) -> tuple[tuple[int, int], int, int]:
    """Return (shape, n_samples, n_center) from the config."""
    size = int(cfg["data"]["image_size"])
    shape = (size, size)
    n_samples = masks.budget_from_fraction(shape, float(cfg["mask"]["sampling_fraction"]))
    n_center = int(round(float(cfg["mask"].get("center_fraction", 0.0)) * n_samples))
    return shape, n_samples, min(n_center, n_samples)


def build_masks(
    names: list[str],
    cfg: dict[str, Any],
    train_images: torch.Tensor,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Build the requested masks by name, sharing one budget and prior."""
    shape, n_samples, n_center = mask_budgets(cfg)
    g = cfg.get("greedy", {})
    noise_var = greedy_noise_var(cfg)

    mask_cfg = cfg.get("mask", {})
    lines_cfg = mask_cfg.get("lines", {})
    ml_cfg = mask_cfg.get("multilevel", {})
    ista_cfg = cfg.get("recon", {}).get("wavelet_ista", {})
    loop_cfg = g.get("recon_in_loop", {})
    n_center_lines = int(lines_cfg.get("n_center_lines", 2))
    wavelet = str(ista_cfg.get("wavelet", "db4"))
    levels = int(ista_cfg.get("levels", 3))
    beta = float(g.get("beta", g.get("artifact_beta", 1.0)))

    def _psf_penalized() -> np.ndarray:
        return greedy.greedy_psf_penalized_aopt(
            _prior(), n_samples,
            noise_var=noise_var,
            beta=beta,
            n_candidates=int(g.get("n_candidates", 32)),
            n_center=n_center,
            rng=rng,
            show_progress=True,
        )

    prior: np.ndarray | None = None
    builders: dict[str, Callable[[], np.ndarray]] = {
        "uniform_random": lambda: masks.uniform_random_mask(shape, n_samples, rng, n_center),
        "variable_density": lambda: masks.variable_density_mask(
            shape, n_samples, rng,
            decay=float(mask_cfg.get("variable_density_decay", 3.0)),
            n_center=n_center,
        ),
        "equispaced_lines": lambda: masks.equispaced_lines_mask(shape, n_samples),
        "variable_density_lines": lambda: masks.variable_density_lines_mask(
            shape, n_samples, rng,
            decay=float(lines_cfg.get("decay", 2.0)),
            n_center_lines=n_center_lines,
        ),
        "multilevel_random": lambda: masks.multilevel_random_mask(
            shape, n_samples, rng,
            n_levels=int(ml_cfg.get("n_levels", 4)),
            decay=float(ml_cfg.get("decay", 1.5)),
        ),
        "aopt_greedy": lambda: greedy.greedy_a_optimal(
            _prior(), n_samples, noise_var=noise_var, n_center=n_center
        ),
        "psf_penalized_aopt_greedy": _psf_penalized,
        # Backward-compatibility alias for the pre-rename mask type name.
        "artifact_aware_greedy": _psf_penalized,
        "data_driven_greedy": lambda: greedy.greedy_data_driven(
            train_images.numpy(), n_samples, n_center=n_center
        ),
        "line_aopt": lambda: greedy.greedy_line_a_optimal(
            _prior(), n_samples, noise_var=noise_var, n_center_lines=n_center_lines
        ),
        "line_subspace_leakage": lambda: greedy.greedy_lines_subspace_leakage(
            train_images.numpy(), n_samples,
            wavelet=wavelet, levels=levels, n_center_lines=n_center_lines,
        ),
        "spectrum_energy_greedy": lambda: greedy.greedy_lines_spectrum_energy(
            train_images.numpy(), n_samples, n_center_lines=n_center_lines
        ),
        "subspace_aopt_greedy": lambda: greedy.greedy_subspace_aoptimal(
            subspace.to_kspace_basis(
                fit_train_subspace(cfg, train_images)[0], shape
            ),
            n_samples,
            sigma2=noise_var,
            n_center=n_center,
            beta=float(cfg.get("subspace", {}).get("beta", 0.0)),
            shape=shape,
            ridge=float(cfg.get("subspace", {}).get("ridge", 1e-6)),
            n_candidates=int(g.get("n_candidates", 32)),
        ),
        "recon_in_loop_greedy": lambda: greedy.greedy_lines_recon_in_loop(
            train_images.numpy(), n_samples,
            n_candidate_lines=int(loop_cfg.get("n_candidate_lines", 12)),
            batch_size=int(loop_cfg.get("batch_size", 8)),
            ista_threshold=float(ista_cfg.get("threshold", 0.02)),
            ista_iters=int(loop_cfg.get("ista_iters", 6)),
            wavelet=wavelet,
            levels=levels,
            n_center_lines=n_center_lines,
            rng=rng,
            show_progress=True,
        ),
    }

    def _prior() -> np.ndarray:
        nonlocal prior
        if prior is None:
            prior = fitted_power_law_spectrum(train_images)
        return prior

    out: dict[str, np.ndarray] = {}
    for name in names:
        if name not in builders:
            raise ValueError(f"unknown mask type {name!r}; known: {sorted(builders)}")
        started = time.monotonic()
        out[name] = builders[name]()
        print(f"  built {name} ({time.monotonic() - started:.1f}s)")
    return out


def build_mask_family(
    cfg: dict[str, Any], train_images: torch.Tensor
) -> dict[str, np.ndarray]:
    """Parameterized family of masks spanning each generator's free parameters.

    Correlating a design-time predictor against measured error over a handful
    of masks has almost no statistical power. This sweeps the parameters that
    actually vary within each family (density decay, level structure, penalty
    weight, random seed) to give the rank correlations a usable sample size.
    Every variant obeys the same budget and center rules as the named masks;
    each random variant gets its own seeded generator so adding a variant
    never perturbs the others.
    """
    shape, n_samples, n_center = mask_budgets(cfg)
    noise_var = greedy_noise_var(cfg)
    mask_cfg = cfg.get("mask", {})
    fam = mask_cfg.get("family", {})
    lines_cfg = mask_cfg.get("lines", {})
    n_center_lines = int(lines_cfg.get("n_center_lines", 2))
    ista_cfg = cfg.get("recon", {}).get("wavelet_ista", {})
    wavelet = str(ista_cfg.get("wavelet", "db4"))
    levels = int(ista_cfg.get("levels", 3))
    n_candidates = int(cfg.get("greedy", {}).get("n_candidates", 32))
    sub_cfg = cfg.get("subspace", {})

    seeds = [int(s) for s in fam.get("seeds", [0, 1, 2])]
    vd_decays = [float(x) for x in fam.get("variable_density_decay", [1.5, 2.5, 3.5, 5.0])]
    vdl_decays = [float(x) for x in fam.get("variable_density_lines_decay", [1.0, 2.0, 4.0])]
    betas = [float(x) for x in fam.get("psf_penalized_beta", [0.5, 1.0, 2.0, 4.0])]
    sub_betas = [float(x) for x in fam.get("subspace_beta", [0.0, 1.0])]
    ml_specs = fam.get(
        "multilevel",
        [
            {"n_levels": 3, "decay": 1.0},
            {"n_levels": 3, "decay": 2.0},
            {"n_levels": 4, "decay": 1.0},
            {"n_levels": 4, "decay": 1.5},
            {"n_levels": 4, "decay": 2.5},
            {"n_levels": 5, "decay": 1.5},
        ],
    )

    cache: dict[str, Any] = {}

    def prior() -> np.ndarray:
        if "prior" not in cache:
            cache["prior"] = fitted_power_law_spectrum(train_images)
        return cache["prior"]

    def kspace_basis() -> np.ndarray:
        if "phi" not in cache:
            basis, _ = fit_train_subspace(cfg, train_images)
            cache["phi"] = subspace.to_kspace_basis(basis, shape)
        return cache["phi"]

    specs: list[tuple[str, Callable[[], np.ndarray]]] = []

    for seed in seeds:
        specs.append(
            (f"uniform_random_s{seed}",
             lambda s=seed: masks.uniform_random_mask(
                 shape, n_samples, np.random.default_rng(s), n_center))
        )
    for decay in vd_decays:
        for seed in seeds[:2]:
            specs.append(
                (f"variable_density_d{decay:g}_s{seed}",
                 lambda d=decay, s=seed: masks.variable_density_mask(
                     shape, n_samples, np.random.default_rng(s), decay=d, n_center=n_center))
            )
    for spec in ml_specs:
        n_levels, decay = int(spec["n_levels"]), float(spec["decay"])
        specs.append(
            (f"multilevel_L{n_levels}_d{decay:g}",
             lambda L=n_levels, d=decay: masks.multilevel_random_mask(
                 shape, n_samples, np.random.default_rng(seeds[0]), n_levels=L, decay=d))
        )
    specs.append(("equispaced_lines", lambda: masks.equispaced_lines_mask(shape, n_samples)))
    for decay in vdl_decays:
        specs.append(
            (f"variable_density_lines_d{decay:g}",
             lambda d=decay: masks.variable_density_lines_mask(
                 shape, n_samples, np.random.default_rng(seeds[0]),
                 decay=d, n_center_lines=n_center_lines))
        )
    specs.append(
        ("aopt_greedy",
         lambda: greedy.greedy_a_optimal(prior(), n_samples, noise_var=noise_var, n_center=n_center))
    )
    for beta in betas:
        specs.append(
            (f"psf_penalized_b{beta:g}",
             lambda b=beta: greedy.greedy_psf_penalized_aopt(
                 prior(), n_samples, noise_var=noise_var, beta=b,
                 n_candidates=n_candidates, n_center=n_center,
                 rng=np.random.default_rng(seeds[0])))
        )
    specs.append(
        ("line_aopt",
         lambda: greedy.greedy_line_a_optimal(
             prior(), n_samples, noise_var=noise_var, n_center_lines=n_center_lines))
    )
    specs.append(
        ("spectrum_energy_greedy",
         lambda: greedy.greedy_lines_spectrum_energy(
             train_images.numpy(), n_samples, n_center_lines=n_center_lines))
    )
    specs.append(
        ("line_subspace_leakage",
         lambda: greedy.greedy_lines_subspace_leakage(
             train_images.numpy(), n_samples,
             wavelet=wavelet, levels=levels, n_center_lines=n_center_lines))
    )
    for beta in sub_betas:
        specs.append(
            (f"subspace_aopt_b{beta:g}",
             lambda b=beta: greedy.greedy_subspace_aoptimal(
                 kspace_basis(), n_samples, sigma2=noise_var, n_center=n_center,
                 beta=b, shape=shape, ridge=float(sub_cfg.get("ridge", 1e-6)),
                 n_candidates=n_candidates))
        )

    out: dict[str, np.ndarray] = {}
    for name, builder in track(specs, total=len(specs), label="build family"):
        out[name] = builder()
    return out


def save_mask_bundle(mask: np.ndarray, name: str, run: Path) -> dict[str, float]:
    """Save mask array, mask image, and PSF plot; return PSF metrics row."""
    _ensure_dir(run / "masks")
    np.save(run / "masks" / f"{name}.npy", mask)
    viz.save_image(mask, run / "masks" / f"{name}.png", title=name)
    viz.plot_psf(mask, run / "psf" / f"{name}_psf.png", title=name)
    row: dict[str, float] = {"mask": name, "n_samples": float(mask.sum())}
    row.update(artifacts.psf_metrics(mask))
    return row


def _ensure_dir(path: Path) -> bool:
    path.mkdir(parents=True, exist_ok=True)
    return True


# ---------------------------------------------------------------------------
# Reconstruction and evaluation
# ---------------------------------------------------------------------------

def reconstruct_all(
    images: torch.Tensor,
    mask: np.ndarray,
    cfg: dict[str, Any],
    generator: torch.Generator | None = None,
    spectrum: np.ndarray | None = None,
    subspace_basis: np.ndarray | None = None,
    gen_model=None,
    gen_z0: torch.Tensor | None = None,
    unet_model=None,
) -> dict[str, torch.Tensor]:
    """Simulate measurements and reconstruct with every configured method.

    zero_filled and wiener are linear diagonal reconstructions: under a
    diagonal prior the estimator's unsampled frequency coefficients stay at
    the prior mean (zero), so both remain confined to the observed subspace
    and their recon_nullspace_norm is ~0. wavelet_ista couples coefficients
    across a non-Fourier basis and can impute null-space content; the
    contrast is intentional.

    spectrum is the mean frequency-domain power estimated on training data
    only. wiener's regularization weight defaults to the simulated measurement
    noise variance (recon.wiener_lambda overrides).

    With subspace_basis (N x d, fitted on training data), the "subspace"
    method solves the prior-constrained least squares in closed form; its
    output lives in the basis span, so it imputes null-space content by
    construction (recon_nullspace_norm > 0 — intentional; see
    recon.subspace_recon). With gen_model and gen_z0, the "generative" method
    optimizes the latent code of a fixed generator per image.

    With unet_model (trained by scripts/12_train_unet.py on training data
    only), the "unet_post" method post-processes the zero-filled magnitude.
    It is a learned prior arm: it imputes null-space content, and — unlike
    wavelet_ista with final_dc — it enforces no data consistency, so its
    consistency_norm is expected to be nonzero. That is a reported property
    of the method, not a defect.
    """
    noise_std = float(cfg.get("measurement", {}).get("noise_std", 0.0))
    y = recon.simulate_measurements(images, mask, noise_std=noise_std, generator=generator)
    out = {"zero_filled": recon.zero_filled(y)}

    if spectrum is None:
        warnings.warn(
            "reconstruct_all: no spectrum provided; wiener falls back to scalar shrinkage",
            stacklevel=2,
        )
        out["wiener"] = recon.ridge(y, mask, float(cfg["recon"]["ridge_lambda"]))
    else:
        lam = float(cfg["recon"].get("wiener_lambda", noise_std**2))
        out["wiener"] = recon.ridge(y, mask, lam, spectrum=spectrum)

    ista_cfg = cfg["recon"].get("wavelet_ista")
    if ista_cfg:
        out["wavelet_ista"] = recon.wavelet_ista(
            y,
            mask,
            threshold=float(ista_cfg["threshold"]),
            n_iters=int(ista_cfg.get("n_iters", 50)),
            wavelet=str(ista_cfg.get("wavelet", "db4")),
            levels=int(ista_cfg.get("levels", 3)),
            final_dc=bool(ista_cfg.get("final_dc", True)),
        )

    if unet_model is not None:
        unet_model.eval()
        with torch.no_grad():
            zf_magnitude = out["zero_filled"].abs().unsqueeze(1)
            out["unet_post"] = unet_model(zf_magnitude).squeeze(1).to(torch.complex64)

    sub_cfg = cfg.get("subspace", {})
    if subspace_basis is not None:
        lam = sub_cfg.get("lam")
        lam = float(lam) if lam is not None else max(noise_std**2, 1e-8)
        out["subspace"] = recon.subspace_recon(y, mask, subspace_basis, lam=lam)
    if gen_model is not None and gen_z0 is not None:
        gen_cfg = sub_cfg.get("generative", {})
        singles = [
            recon.generative_recon(
                y[i], mask, gen_model, gen_z0,
                steps=int(gen_cfg.get("steps", 200)),
                lr=float(gen_cfg.get("lr", 0.05)),
            )
            for i in range(y.shape[0])
        ]
        out["generative"] = torch.stack(singles)
    return out


def metrics_rows(
    recons: dict[str, torch.Tensor],
    truth: torch.Tensor,
    mask: np.ndarray,
    mask_name: str,
) -> list[dict[str, Any]]:
    """Per-image metric rows for every method, including artifact metrics.

    Alongside the magnitude-based scalar metrics, each row carries the
    observed-subspace / null-space error decomposition norms computed on the
    complex reconstruction (before taking magnitudes).
    """
    rows = []
    for method, batch in recons.items():
        for i in range(truth.shape[0]):
            truth_i = truth[i].numpy()
            recon_i = batch[i].abs().numpy()
            row: dict[str, Any] = {"mask": mask_name, "method": method, "image_index": i}
            row.update(metrics.evaluate(recon_i, truth_i))
            row["aliasing_energy_ratio"] = artifacts.aliasing_energy_ratio(truth[i], mask)
            dec = artifacts.decompose_error(batch[i], truth[i], mask)
            row["artifact_norm"] = dec.artifact_norm
            row["consistency_norm"] = dec.consistency_norm
            row["recon_nullspace_norm"] = dec.recon_nullspace_norm
            row["truth_nullspace_norm"] = dec.truth_nullspace_norm
            row["no_nullspace_content"] = dec.no_nullspace_content
            rows.append(row)
    return rows


def save_examples(
    truth: torch.Tensor,
    recons: dict[str, torch.Tensor],
    mask: np.ndarray,
    mask_name: str,
    run: Path,
    indices: list[int],
) -> None:
    """Save reconstruction, total-error, artifact-field, and null-space grids.

    Total error |recon - truth| goes to artifact_maps/<mask>_<method>.png as
    before; the null-space imputation error magnitude |artifact_field| and the
    reconstruction's invented null-space content |(I - P) recon| get the
    suffixes _artifact_field.png and _nullspace.png.
    """
    for method, batch in recons.items():
        recon_images = [batch[i].abs().numpy() for i in indices]
        error_maps = [artifacts.artifact_map(batch[i], truth[i]).numpy() for i in indices]
        decs = [artifacts.decompose_error(batch[i], truth[i], mask) for i in indices]
        titles = [f"test[{i}]" for i in indices]
        viz.save_image_grid(
            recon_images,
            run / "recon" / f"{mask_name}_{method}.png",
            titles=titles,
            vmin=0.0,
            vmax=1.0,
        )
        viz.save_image_grid(
            error_maps,
            run / "artifact_maps" / f"{mask_name}_{method}.png",
            titles=titles,
            cmap="inferno",
        )
        viz.save_image_grid(
            [dec.artifact_field.abs().numpy() for dec in decs],
            run / "artifact_maps" / f"{mask_name}_{method}_artifact_field.png",
            titles=titles,
            cmap="inferno",
        )
        viz.save_image_grid(
            [dec.recon_nullspace.abs().numpy() for dec in decs],
            run / "artifact_maps" / f"{mask_name}_{method}_nullspace.png",
            titles=titles,
            cmap="inferno",
        )


def evaluate_masks(
    mask_dict: dict[str, np.ndarray],
    test_images: torch.Tensor,
    cfg: dict[str, Any],
    run: Path,
    prefix: str,
    example_indices: list[int] | None = None,
    spectrum: np.ndarray | None = None,
    subspace_basis: np.ndarray | None = None,
    unet_model=None,
    write_examples: bool = True,
) -> pd.DataFrame:
    """Full evaluation of a set of masks: metrics CSV, examples, PSF metrics.

    Returns the per-image metrics DataFrame (also written to
    runs/<exp>/metrics/<prefix>_metrics.csv).
    """
    generator = torch.Generator().manual_seed(int(cfg["seed"]))
    n_examples = int(cfg.get("outputs", {}).get("n_examples", 5))

    all_rows: list[dict[str, Any]] = []
    psf_rows: list[dict[str, float]] = []
    recons_by_mask: dict[str, dict[str, torch.Tensor]] = {}
    for name in track(mask_dict, total=len(mask_dict), label=f"evaluate[{prefix}]"):
        mask = mask_dict[name]
        psf_rows.append(save_mask_bundle(mask, name, run))
        recons = reconstruct_all(
            test_images, mask, cfg,
            generator=generator, spectrum=spectrum, subspace_basis=subspace_basis,
            unet_model=unet_model,
        )
        recons_by_mask[name] = recons
        all_rows.extend(metrics_rows(recons, test_images, mask, name))

    frame = pd.DataFrame(all_rows)
    # Each mask/method pair costs three rendered grids; a parameter sweep over
    # dozens of masks spends more time in matplotlib than in reconstruction.
    if write_examples:
        if example_indices is None:
            example_indices = representative_indices(frame, n_examples, test_images.shape[0])
        for name, mask in mask_dict.items():
            save_examples(test_images, recons_by_mask[name], mask, name, run, example_indices)

    _ensure_dir(run / "metrics")
    frame.to_csv(run / "metrics" / f"{prefix}_metrics.csv", index=False)
    pd.DataFrame(psf_rows).to_csv(run / "metrics" / f"{prefix}_psf_metrics.csv", index=False)
    return frame


def argumentation_table(
    mask_dict: dict[str, np.ndarray],
    frame: pd.DataFrame,
    train_power: np.ndarray,
    mass: dict[str, np.ndarray] | None = None,
    energies: dict[str, float] | None = None,
    basis: np.ndarray | None = None,
    baseline_method: str = "zero_filled",
) -> pd.DataFrame:
    """Design-time predictors next to measured outcomes, one row per mask.

    The point is argumentative rather than descriptive: every predictor claims
    to rank masks before any measurement is simulated, and the outcome columns
    test that claim. Per-method mean MSE columns (mse_<method>) and PSNR gains
    over the baseline method (psnr_gain_<method>) are emitted for whichever
    methods are present in `frame`, so learned and iterative arms join the
    correlation test automatically.
    """
    rows = []
    for name, mask in mask_dict.items():
        sub = frame[frame["mask"] == name]
        base = sub[sub["method"] == baseline_method]
        row: dict[str, Any] = {"mask": name}
        row["mask_score"] = artifacts.expected_zero_filled_mse(mask, train_power)
        row.update(artifacts.psf_metrics(mask))
        row.update(artifacts.spectrum_weighted_psf_metrics(mask, train_power))
        if mass is not None and energies is not None:
            row["wavelet_leakage"] = artifacts.wavelet_leakage_score(mask, mass, energies)
        if basis is not None:
            row["subspace_leakage"] = artifacts.subspace_nullspace_leakage(basis, mask)
        row["truth_nullspace_norm"] = float(base["truth_nullspace_norm"].mean())
        for method in sorted(sub["method"].unique()):
            block = sub[sub["method"] == method]
            row[f"mse_{method}"] = float(block["mse"].mean())
            row[f"nullspace_{method}"] = float(block["recon_nullspace_norm"].mean())
            row[f"consistency_{method}"] = float(block["consistency_norm"].mean())
            if method != baseline_method:
                row[f"psnr_gain_{method}"] = float(
                    block["psnr"].mean() - base["psnr"].mean()
                )
        rows.append(row)
    return pd.DataFrame(rows)


def rank_correlations(
    table: pd.DataFrame, predictors: list[str], outcomes: list[str]
) -> pd.DataFrame:
    """Spearman rank correlation of each predictor against each outcome."""
    from scipy.stats import spearmanr

    rows = []
    for predictor in predictors:
        if predictor not in table:
            continue
        for outcome in outcomes:
            if outcome not in table or table[outcome].isna().all():
                continue
            if table[predictor].nunique() < 2 or table[outcome].nunique() < 2:
                continue
            rho, p_value = spearmanr(table[predictor], table[outcome])
            rows.append(
                {
                    "predictor": predictor,
                    "outcome": outcome,
                    "n_masks": int(len(table)),
                    "spearman_rho": float(rho),
                    "p_value": float(p_value),
                }
            )
    return pd.DataFrame(rows)


def representative_indices(frame: pd.DataFrame, n_examples: int, n_test: int) -> list[int]:
    """Test indices spread across the difficulty range (quantiles of mean NRMSE)."""
    per_image = frame.groupby("image_index")["nrmse"].mean().reindex(range(n_test))
    order = per_image.sort_values().index.to_numpy()
    positions = np.unique(np.linspace(0, len(order) - 1, n_examples).astype(int))
    return sorted(int(order[p]) for p in positions)
