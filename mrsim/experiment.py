"""Shared experiment plumbing used by the numbered scripts."""

from __future__ import annotations

import hashlib
import json
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch

from . import artifacts, data, greedy, masks, metrics, recon, subspace, viz
from .config import config_fingerprint
from .fft_ops import fft2c
from .progress import track


# ---------------------------------------------------------------------------
# Dataset handling
# ---------------------------------------------------------------------------

def dataset_path(run: Path) -> Path:
    return run / "data" / "dataset.pt"


def _dataset_metadata(cfg: dict[str, Any]) -> dict[str, Any]:
    d = cfg["data"]
    source = str(d.get("source", "synthetic"))
    metadata = {
        "source": source,
        "seed": int(cfg["seed"]),
        "n_images": int(d["n_images"]),
        "image_size": int(d["image_size"]),
    }
    if source == "synthetic":
        metadata.update(
            {
                "phantom": str(d.get("phantom", "ellipses")),
                "min_ellipses": int(d.get("min_ellipses", 3)),
                "max_ellipses": int(d.get("max_ellipses", 8)),
            }
        )
        return metadata

    source_path = Path(d["path"])
    metadata.update(
        {
            "path": str(source_path.resolve()),
            "sha256": _file_sha256(source_path),
        }
    )
    return metadata


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _can_refresh_dataset_cache(
    cached_metadata: dict[str, Any], expected_metadata: dict[str, Any]
) -> bool:
    cached_source = str(cached_metadata.get("source", "synthetic"))
    expected_source = str(expected_metadata.get("source", "synthetic"))
    if cached_source != expected_source:
        return True
    return expected_source != "synthetic"


def load_or_generate_dataset(cfg: dict[str, Any], run: Path, force: bool = False) -> torch.Tensor:
    """Load the run's dataset, rejecting stale or incompatible cache entries."""
    path = dataset_path(run)
    expected_metadata = _dataset_metadata(cfg)
    if path.exists() and not force:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or "images" not in payload or "metadata" not in payload:
            raise ValueError(
                f"{path} is a legacy dataset cache without metadata; regenerate it with script 01"
            )
        cached_metadata = payload["metadata"]
        if cached_metadata == expected_metadata:
            return payload["images"]
        if not _can_refresh_dataset_cache(cached_metadata, expected_metadata):
            raise ValueError(
                f"{path} was generated from a different data configuration; "
                "use the matching config-hash run or regenerate it"
            )
    d = cfg["data"]
    if str(d.get("source", "synthetic")) == "synthetic":
        images = data.generate_dataset(
            n_images=int(d["n_images"]),
            size=int(d["image_size"]),
            seed=int(cfg["seed"]),
            phantom=str(d.get("phantom", "ellipses")),
            min_ellipses=int(d.get("min_ellipses", 3)),
            max_ellipses=int(d.get("max_ellipses", 8)),
        )
    else:
        images = data.load_array_dataset(
            d["path"], int(d["n_images"]), int(d["image_size"])
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"images": images, "metadata": expected_metadata}, path)
    return images


def train_validation_test_split(
    images: torch.Tensor, cfg: dict[str, Any]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Deterministic, disjoint train/validation/test split."""
    d = cfg["data"]
    n_train = int(d["n_train"])
    n_val = int(d.get("n_val", 0))
    n_test = int(d["n_test"])
    if n_train + n_val + n_test > images.shape[0]:
        raise ValueError("n_train + n_val + n_test exceeds the dataset size")
    val_start = n_train
    test_start = n_train + n_val
    return (
        images[:n_train],
        images[val_start:test_start],
        images[test_start : test_start + n_test],
    )


def train_test_split(images: torch.Tensor, cfg: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    """Compatibility wrapper returning train and the held-out test split."""
    train, _, test = train_validation_test_split(images, cfg)
    return train, test


train_val_test_split = train_validation_test_split


# ---------------------------------------------------------------------------
# Spectra and mask construction
# ---------------------------------------------------------------------------

def mean_power_spectrum(images: torch.Tensor) -> np.ndarray:
    """Second moment E|X_k|^2 over a stack of images."""
    k = fft2c(images.to(torch.complex64))
    return (k.abs() ** 2).mean(dim=0).detach().cpu().numpy().astype(np.float64)


def frequency_prior_statistics(
    images: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return complex mean, centered variance, and second moment in k-space."""
    k = fft2c(images.to(torch.complex64))
    mean = k.mean(dim=0)
    variance = ((k - mean).abs() ** 2).mean(dim=0)
    second_moment = (k.abs() ** 2).mean(dim=0)
    return (
        mean.detach().cpu().numpy().astype(np.complex64),
        variance.detach().cpu().numpy().astype(np.float64),
        second_moment.detach().cpu().numpy().astype(np.float64),
    )


def fitted_power_law_spectrum(images: torch.Tensor) -> np.ndarray:
    """Radially symmetric power-law fit to the centered spectral variance.

    Fits log s = intercept + slope * log(1 + r) by least squares and rebuilds
    a smooth spectrum from the radius map. Used as the model prior for
    A-optimal and artifact-aware greedy selection.
    """
    _, empirical, _ = frequency_prior_statistics(images)
    r = masks.radius_map(empirical.shape)
    x = np.log1p(r).ravel()
    y = np.log(empirical.ravel() + 1e-12)
    slope, intercept = np.polyfit(x, y, 1)
    spectrum = np.exp(intercept) * (1.0 + r) ** slope
    return np.maximum(spectrum, 1e-12)


def load_unet(run: Path, cfg: dict[str, Any] | None = None):
    """Load a compatible experimental U-Net checkpoint, when present."""
    path = run / "models" / "unet_post.pt"
    if not path.exists():
        return None
    from .unet import SmallUNet

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if cfg is not None:
        expected_size = [int(cfg["data"]["image_size"])] * 2
        if payload.get("image_size") != expected_size:
            warnings.warn(
                f"ignoring {path}: checkpoint image size "
                f"{payload.get('image_size')} != {expected_size}",
                stacklevel=2,
            )
            return None
        expected_fingerprint = config_fingerprint(cfg)
        if payload.get("config_fingerprint") != expected_fingerprint:
            warnings.warn(
                f"ignoring {path}: checkpoint was trained under another config",
                stacklevel=2,
            )
            return None
    model = SmallUNet(base_channels=int(payload.get("base_channels", 16)))
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def fit_train_subspace(
    cfg: dict[str, Any], train_images: torch.Tensor
) -> tuple[np.ndarray, float]:
    """Fit a centered config-sized subspace basis on the train split."""
    statistics = fit_train_subspace_statistics(cfg, train_images)
    return statistics.basis, statistics.energy_ratio


def fit_train_subspace_statistics(
    cfg: dict[str, Any], train_images: torch.Tensor
) -> subspace.SubspaceStatistics:
    """Fit the empirical mean and covariance subspace on training data."""
    d = int(cfg.get("subspace", {}).get("d", 32))
    statistics = subspace.fit_subspace(
        train_images,
        d,
        center=True,
        return_statistics=True,
    )
    if not isinstance(statistics, subspace.SubspaceStatistics):
        raise TypeError("centered subspace fit did not return statistics")
    return statistics


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


def subspace_regularization(cfg: dict[str, Any]) -> float:
    """Noise-matched affine-Gaussian regularization with a numerical floor."""
    sub_cfg = cfg.get("subspace", {})
    floor = float(sub_cfg.get("ridge", 1e-8))
    configured = sub_cfg.get("lam")
    measurement_variance = float(
        cfg.get("measurement", {}).get("noise_std", 0.0)
    ) ** 2
    value = measurement_variance if configured is None else float(configured)
    if not np.isfinite(floor) or floor <= 0.0:
        raise ValueError("subspace.ridge must be finite and positive")
    if not np.isfinite(value) or value < 0.0:
        raise ValueError("subspace.lam must be finite and non-negative")
    return max(value, floor)


def mask_budgets(cfg: dict[str, Any]) -> tuple[tuple[int, int], int, int]:
    """Return (shape, n_samples, n_center) from the config."""
    size = int(cfg["data"]["image_size"])
    shape = (size, size)
    n_samples = masks.budget_from_fraction(shape, float(cfg["mask"]["sampling_fraction"]))
    n_center = int(round(float(cfg["mask"].get("center_fraction", 0.0)) * n_samples))
    return shape, n_samples, min(n_center, n_samples)


def _named_rng(seed: int, *parts: str) -> np.random.Generator:
    """Order-independent RNG derived from a global seed and stable names."""
    payload = "\0".join((str(seed), *parts)).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    child_seed = int.from_bytes(digest[:8], "little", signed=False)
    return np.random.default_rng(child_seed)


def _to_numpy(images: torch.Tensor) -> np.ndarray:
    """Detach a tensor for NumPy-only design routines."""
    return images.detach().cpu().numpy()


def build_masks(
    names: list[str],
    cfg: dict[str, Any],
    train_images: torch.Tensor,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Build requested masks with order-independent per-mask randomness.

    ``rng`` remains in the public signature for compatibility. Random mask
    streams are intentionally derived from the config seed and mask name so
    adding or reordering another mask cannot perturb existing results.
    """
    shape, n_samples, n_center = mask_budgets(cfg)
    g = cfg.get("greedy", {})
    noise_var = greedy_noise_var(cfg)
    seed = int(cfg["seed"])

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
            rng=_named_rng(seed, "mask", "psf_penalized_aopt_greedy"),
            show_progress=True,
        )

    prior: np.ndarray | None = None
    fitted_subspace: subspace.SubspaceStatistics | None = None
    builders: dict[str, Callable[[], np.ndarray]] = {
        "uniform_random": lambda: masks.uniform_random_mask(
            shape, n_samples, _named_rng(seed, "mask", "uniform_random"), n_center
        ),
        "variable_density": lambda: masks.variable_density_mask(
            shape, n_samples, _named_rng(seed, "mask", "variable_density"),
            decay=float(mask_cfg.get("variable_density_decay", 3.0)),
            n_center=n_center,
        ),
        "equispaced_lines": lambda: masks.equispaced_lines_mask(shape, n_samples),
        "variable_density_lines": lambda: masks.variable_density_lines_mask(
            shape, n_samples, _named_rng(seed, "mask", "variable_density_lines"),
            decay=float(lines_cfg.get("decay", 2.0)),
            n_center_lines=n_center_lines,
        ),
        "multilevel_random": lambda: masks.multilevel_random_mask(
            shape, n_samples, _named_rng(seed, "mask", "multilevel_random"),
            n_levels=int(ml_cfg.get("n_levels", 4)),
            decay=float(ml_cfg.get("decay", 1.5)),
            n_center=n_center,
        ),
        "aopt_greedy": lambda: greedy.greedy_a_optimal(
            _prior(), n_samples, noise_var=noise_var, n_center=n_center
        ),
        "psf_penalized_aopt_greedy": _psf_penalized,
        # Backward-compatibility alias for the pre-rename mask type name.
        "artifact_aware_greedy": _psf_penalized,
        "data_driven_greedy": lambda: greedy.greedy_data_driven(
            _to_numpy(train_images), n_samples, n_center=n_center
        ),
        "line_aopt": lambda: greedy.greedy_line_a_optimal(
            _prior(), n_samples, noise_var=noise_var, n_center_lines=n_center_lines
        ),
        "line_subspace_leakage": lambda: greedy.greedy_lines_subspace_leakage(
            _to_numpy(train_images), n_samples,
            wavelet=wavelet, levels=levels, n_center_lines=n_center_lines,
        ),
        "spectrum_energy_greedy": lambda: greedy.greedy_lines_spectrum_energy(
            _to_numpy(train_images), n_samples, n_center_lines=n_center_lines
        ),
        "subspace_aopt_greedy": lambda: greedy.greedy_subspace_aoptimal(
            subspace.to_kspace_basis(
                _subspace_statistics().basis,
                shape,
            ),
            n_samples,
            sigma2=subspace_regularization(cfg),
            prior_variances=_subspace_statistics().eigenvalues,
            n_center=n_center,
            beta=float(cfg.get("subspace", {}).get("beta", 0.0)),
            shape=shape,
            ridge=subspace_regularization(cfg),
            n_candidates=int(g.get("n_candidates", 32)),
        ),
        "recon_in_loop_greedy": lambda: greedy.greedy_lines_recon_in_loop(
            _to_numpy(train_images), n_samples,
            n_candidate_lines=int(loop_cfg.get("n_candidate_lines", 12)),
            batch_size=int(loop_cfg.get("batch_size", 8)),
            ista_threshold=float(ista_cfg.get("threshold", 0.02)),
            ista_iters=int(loop_cfg.get("ista_iters", 6)),
            wavelet=wavelet,
            levels=levels,
            n_center_lines=n_center_lines,
            rng=_named_rng(seed, "mask", "recon_in_loop_greedy"),
            show_progress=True,
        ),
    }

    def _prior() -> np.ndarray:
        nonlocal prior
        if prior is None:
            prior = fitted_power_law_spectrum(train_images)
        return prior

    def _subspace_statistics() -> subspace.SubspaceStatistics:
        nonlocal fitted_subspace
        if fitted_subspace is None:
            fitted_subspace = fit_train_subspace_statistics(cfg, train_images)
        return fitted_subspace

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
    cache_lock = threading.Lock()

    # Guarded so that with n_workers > 1 several threads cannot enter the
    # cache-miss branch at once and redundantly recompute the fitted spectrum
    # or the subspace SVD.
    def prior() -> np.ndarray:
        with cache_lock:
            if "prior" not in cache:
                cache["prior"] = fitted_power_law_spectrum(train_images)
            return cache["prior"]

    def kspace_basis() -> np.ndarray:
        if "phi" not in cache:
            statistics = fit_train_subspace_statistics(cfg, train_images)
            cache["statistics"] = statistics
            cache["phi"] = subspace.to_kspace_basis(statistics.basis, shape)
        return cache["phi"]

    def subspace_variances() -> np.ndarray:
        kspace_basis()
        return cache["statistics"].eigenvalues

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
                 shape,
                 n_samples,
                 np.random.default_rng(seeds[0]),
                 n_levels=L,
                 decay=d,
                 n_center=n_center,
             ))
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
             _to_numpy(train_images), n_samples, n_center_lines=n_center_lines))
    )
    specs.append(
        ("line_subspace_leakage",
         lambda: greedy.greedy_lines_subspace_leakage(
             _to_numpy(train_images), n_samples,
             wavelet=wavelet, levels=levels, n_center_lines=n_center_lines))
    )
    for beta in sub_betas:
        specs.append(
            (f"subspace_aopt_b{beta:g}",
             lambda b=beta: greedy.greedy_subspace_aoptimal(
                 kspace_basis(),
                 n_samples,
                 sigma2=subspace_regularization(cfg),
                 prior_variances=subspace_variances(),
                 n_center=n_center,
                 beta=b,
                 shape=shape,
                 ridge=subspace_regularization(cfg),
                 n_candidates=n_candidates))
        )

    out: dict[str, np.ndarray] = {}
    workers = int(cfg.get("n_workers", 1) or 1)
    if workers > 1:
        # Builders are independent, and NumPy's FFT releases the GIL, so a
        # thread pool gives real parallelism. Processes are not an option: the
        # builders are lambdas and cannot be pickled.
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(builder): name for name, builder in specs}
            for future in track(
                as_completed(futures), total=len(specs), label="build family"
            ):
                out[futures[future]] = future.result()
        # Restore spec order so results never depend on completion order.
        return {name: out[name] for name, _ in specs}

    for name, builder in track(specs, total=len(specs), label="build family"):
        out[name] = builder()
    return out


def save_mask_bundle(mask: np.ndarray, name: str, run: Path) -> dict[str, float]:
    """Validate and save a mask plus an explicit artifact manifest entry."""
    mask = np.asarray(mask)
    if mask.ndim != 2:
        raise ValueError(f"mask {name!r} must be two-dimensional, got shape {mask.shape}")
    if not np.isin(mask, (0.0, 1.0)).all():
        raise ValueError(f"mask {name!r} must be binary")
    _ensure_dir(run / "masks")
    np.save(run / "masks" / f"{name}.npy", mask)
    manifest_path = run / "masks" / "manifest.json"
    manifest = {"version": 1, "masks": {}}
    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as handle:
            loaded = json.load(handle)
        if isinstance(loaded, dict) and isinstance(loaded.get("masks"), dict):
            manifest = loaded
    manifest["masks"][name] = {
        "path": f"{name}.npy",
        "shape": list(mask.shape),
        "n_samples": int(mask.sum()),
        "sha256": hashlib.sha256(np.ascontiguousarray(mask).tobytes()).hexdigest(),
    }
    temp_path = manifest_path.with_suffix(".json.tmp")
    with open(temp_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    temp_path.replace(manifest_path)
    viz.save_image(mask, run / "masks" / f"{name}.png", title=name)
    viz.plot_psf(mask, run / "psf" / f"{name}_psf.png", title=name)
    row: dict[str, float] = {"mask": name, "n_samples": float(mask.sum())}
    row.update(artifacts.psf_metrics(mask))
    return row


def _ensure_dir(path: Path) -> bool:
    path.mkdir(parents=True, exist_ok=True)
    return True


def acquisition_family(mask: np.ndarray) -> str:
    """Classify a binary mask by its realized acquisition geometry."""
    mask_array = np.asarray(mask)
    return (
        "full_cartesian_lines"
        if mask_array.ndim == 2
        and np.all(mask_array == mask_array[0:1, :])
        else "point_sampling"
    )


# ---------------------------------------------------------------------------
# Reconstruction and evaluation
# ---------------------------------------------------------------------------

def reconstruct_all(
    images: torch.Tensor,
    mask: np.ndarray,
    cfg: dict[str, Any],
    generator: torch.Generator | None = None,
    noise: np.ndarray | torch.Tensor | None = None,
    spectrum: np.ndarray | None = None,
    prior_mean: np.ndarray | None = None,
    subspace_basis: np.ndarray | None = None,
    subspace_mean: np.ndarray | None = None,
    subspace_variances: np.ndarray | None = None,
    gen_model=None,
    gen_z0: torch.Tensor | None = None,
    unet_model=None,
    ista_threshold: float | None = None,
    return_measurements: bool = False,
) -> dict[str, torch.Tensor] | tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Simulate measurements and reconstruct with every configured method.

    Zero filling is confined to the observed subspace. Wiener is a diagonal
    Gaussian posterior mean: with the empirical ``prior_mean`` used by the
    main pipeline, it fills unmeasured coefficients with that mean and is
    therefore explicitly prior-dependent. Wavelet ISTA couples coefficients
    across a non-Fourier basis and can also impute null-space content.

    ``spectrum`` is the centered frequency-domain variance estimated on
    training data only. ``prior_mean`` is the corresponding complex mean.
    Wiener's regularization weight defaults to the simulated measurement noise
    variance (``recon.wiener_lambda`` overrides).

    ``noise`` supplies an explicit full-grid realization. Reusing the same
    tensor across masks makes a comparison paired and independent of mask
    iteration order. When ``return_measurements`` is true, the masked
    measurements are returned with the reconstructions for residual metrics.

    With subspace_basis (N x d, fitted on training data), the "subspace"
    method solves the affine Gaussian posterior mean in closed form, using
    ``subspace_mean`` and ``subspace_variances`` when supplied. Its output can
    impute null-space content; see ``recon.subspace_recon``. With
    ``gen_model`` and ``gen_z0``, the "generative" method optimizes a latent
    code per image.

    With unet_model (trained by scripts/12_train_unet.py on training data
    only), the "unet_post" method post-processes the zero-filled magnitude.
    It is a learned prior arm: it imputes null-space content and enforces no
    hard data consistency, so its observable measurement residual can be
    nonzero. That is a reported property of the method, not a certificate.
    """
    noise_std = float(cfg.get("measurement", {}).get("noise_std", 0.0))
    if noise is None:
        y = recon.simulate_measurements(
            images, mask, noise_std=noise_std, generator=generator
        )
    else:
        y = recon.simulate_measurements(images, mask, noise=noise)
    out = {"zero_filled": recon.zero_filled(y)}

    if spectrum is None:
        warnings.warn(
            "reconstruct_all: no spectrum provided; wiener falls back to scalar shrinkage",
            stacklevel=2,
        )
        out["wiener"] = recon.ridge(
            y,
            mask,
            float(cfg["recon"]["ridge_lambda"]),
            prior_mean=prior_mean,
        )
    else:
        lam = float(cfg["recon"].get("wiener_lambda", noise_std**2))
        out["wiener"] = recon.ridge(
            y,
            mask,
            lam,
            spectrum=spectrum,
            prior_mean=prior_mean,
        )

    ista_cfg = cfg["recon"].get("wavelet_ista")
    if ista_cfg:
        def run_ista(threshold: float) -> torch.Tensor:
            return recon.wavelet_ista(
                y,
                mask,
                threshold=threshold,
                n_iters=int(ista_cfg.get("n_iters", 50)),
                wavelet=str(ista_cfg.get("wavelet", "db4")),
                levels=int(ista_cfg.get("levels", 3)),
                final_dc=bool(ista_cfg.get("final_dc", True)),
            )

        fixed = float(ista_cfg["threshold"])
        out["wavelet_ista"] = run_ista(fixed if ista_threshold is None else ista_threshold)
        # Keep the fixed-threshold arm alongside the tuned one so the gap
        # between "best achievable ISTA for this mask" and "one global
        # threshold" is visible rather than assumed away.
        if ista_threshold is not None and not np.isclose(ista_threshold, fixed):
            out["wavelet_ista_fixed"] = run_ista(fixed)

    if unet_model is not None:
        unet_model.eval()
        with torch.no_grad():
            parameter = next(unet_model.parameters())
            zf_magnitude = out["zero_filled"].abs().unsqueeze(1).to(
                device=parameter.device,
                dtype=parameter.dtype,
            )
            prediction = unet_model(zf_magnitude).squeeze(1)
            out["unet_post"] = prediction.to(images.device).to(torch.complex64)

    sub_cfg = cfg.get("subspace", {})
    if subspace_basis is not None:
        lam = subspace_regularization(cfg)
        out["subspace"] = recon.subspace_recon(
            y,
            mask,
            subspace_basis,
            lam=lam,
            prior_mean=subspace_mean,
            coefficient_variances=subspace_variances,
        )
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
    if return_measurements:
        return out, y
    return out


def metrics_rows(
    recons: dict[str, torch.Tensor],
    truth: torch.Tensor,
    mask: np.ndarray,
    mask_name: str,
    measurements: torch.Tensor | None = None,
) -> list[dict[str, Any]]:
    """Per-image metric rows for every method, including artifact metrics.

    Alongside the magnitude-based scalar metrics, each row carries the
    observed-subspace / null-space error decomposition norms computed on the
    complex reconstruction (before taking magnitudes).
    """
    rows = []
    for method, batch in recons.items():
        for i in range(truth.shape[0]):
            truth_i = truth[i].detach().cpu().numpy()
            recon_i = batch[i].detach().cpu().numpy()
            row: dict[str, Any] = {"mask": mask_name, "method": method, "image_index": i}
            quality = metrics.evaluate(recon_i, truth_i)
            quality.pop("mse")  # CSVs use only the explicit MSE names below.
            row.update(quality)
            row["magnitude_mse"] = metrics.magnitude_mse(recon_i, truth_i)
            row["complex_mse"] = metrics.complex_mse(recon_i, truth_i)
            row["oracle_unsampled_energy_ratio"] = artifacts.aliasing_energy_ratio(
                truth[i], mask
            )
            dec = artifacts.decompose_error(batch[i], truth[i], mask)
            row["oracle_nullspace_error_norm"] = dec.artifact_norm
            row["oracle_observed_subspace_error_norm"] = (
                dec.observed_subspace_error_norm
            )
            row["recon_nullspace_norm"] = dec.recon_nullspace_norm
            row["oracle_truth_nullspace_norm"] = dec.truth_nullspace_norm
            row["no_reconstructed_nullspace_content"] = dec.no_nullspace_content
            if measurements is not None:
                row["measurement_residual_norm"] = artifacts.measurement_residual_norm(
                    batch[i], measurements[i], mask
                )
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
        recon_images = [batch[i].abs().detach().cpu().numpy() for i in indices]
        error_maps = [
            artifacts.artifact_map(batch[i], truth[i]).detach().cpu().numpy()
            for i in indices
        ]
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
            [dec.artifact_field.abs().detach().cpu().numpy() for dec in decs],
            run / "artifact_maps" / f"{mask_name}_{method}_artifact_field.png",
            titles=titles,
            cmap="inferno",
        )
        viz.save_image_grid(
            [dec.recon_nullspace.abs().detach().cpu().numpy() for dec in decs],
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
    prior_mean: np.ndarray | None = None,
    subspace_basis: np.ndarray | None = None,
    subspace_mean: np.ndarray | None = None,
    subspace_variances: np.ndarray | None = None,
    unet_model=None,
    write_examples: bool = True,
    val_images: torch.Tensor | None = None,
) -> pd.DataFrame:
    """Full evaluation of a set of masks: metrics CSV, examples, PSF metrics.

    Returns the per-image metrics DataFrame (also written to
    runs/<exp>/metrics/<prefix>_metrics.csv).
    """
    tune_ista = val_images is not None and cfg.get("recon", {}).get("wavelet_ista")
    if tune_ista and val_images.shape[0] == 0:
        raise ValueError("ISTA threshold tuning requires a non-empty validation split")
    n_examples = int(cfg.get("outputs", {}).get("n_examples", 5))
    noise_std = float(cfg.get("measurement", {}).get("noise_std", 0.0))
    common_noise = None
    if noise_std > 0.0:
        common_noise = recon.sample_complex_noise_like(
            test_images,
            noise_std,
            generator=torch.Generator().manual_seed(int(cfg["seed"]) + 701),
        )
    if write_examples and example_indices is None:
        n_selected = min(n_examples, int(test_images.shape[0]))
        example_indices = np.unique(
            np.linspace(0, test_images.shape[0] - 1, n_selected).astype(int)
        ).tolist()
    if example_indices is not None and any(
        index < 0 or index >= test_images.shape[0] for index in example_indices
    ):
        raise IndexError("example_indices contains an index outside the test split")

    all_rows: list[dict[str, Any]] = []
    psf_rows: list[dict[str, float]] = []
    ista_threshold_tables: list[pd.DataFrame] = []
    for name in track(mask_dict, total=len(mask_dict), label=f"evaluate[{prefix}]"):
        mask = np.asarray(mask_dict[name])
        if tuple(mask.shape) != tuple(test_images.shape[-2:]):
            raise ValueError(
                f"mask {name!r} shape {mask.shape} does not match test image "
                f"shape {tuple(test_images.shape[-2:])}"
            )
        psf_rows.append(save_mask_bundle(mask, name, run))
        selected_ista_threshold = None
        if tune_ista:
            selected_ista_threshold, threshold_table = tune_ista_threshold(
                val_images, mask, cfg
            )
            threshold_table = threshold_table.copy()
            threshold_table.insert(0, "mask", name)
            threshold_table["selected"] = (
                threshold_table["threshold"] == selected_ista_threshold
            )
            ista_threshold_tables.append(threshold_table)
        reconstructed = reconstruct_all(
            test_images, mask, cfg,
            noise=common_noise,
            spectrum=spectrum,
            prior_mean=prior_mean,
            subspace_basis=subspace_basis,
            subspace_mean=subspace_mean,
            subspace_variances=subspace_variances,
            unet_model=unet_model,
            ista_threshold=selected_ista_threshold,
            return_measurements=True,
        )
        recons, measurements = reconstructed
        all_rows.extend(
            metrics_rows(
                recons,
                test_images,
                mask,
                name,
                measurements=measurements,
            )
        )
        if write_examples and example_indices:
            save_examples(test_images, recons, mask, name, run, example_indices)

    frame = pd.DataFrame(all_rows)

    _ensure_dir(run / "metrics")
    frame.to_csv(run / "metrics" / f"{prefix}_metrics.csv", index=False)
    pd.DataFrame(psf_rows).to_csv(run / "metrics" / f"{prefix}_psf_metrics.csv", index=False)
    if ista_threshold_tables:
        pd.concat(ista_threshold_tables, ignore_index=True).to_csv(
            run / "metrics" / f"{prefix}_ista_thresholds.csv", index=False
        )
    return frame


DEFAULT_THRESHOLD_GRID = [0.0025, 0.005, 0.01, 0.02, 0.04, 0.08]


def tune_ista_threshold(
    val_images: torch.Tensor,
    mask: np.ndarray,
    cfg: dict[str, Any],
) -> tuple[float, pd.DataFrame]:
    """Pick the wavelet-ISTA threshold for one mask on the validation split.

    A single global threshold confounds mask comparison: a mask whose
    zero-filled reconstruction is already near-optimal has little aliasing to
    remove, so a threshold tuned for a harder mask only adds bias there and can
    turn the measured ISTA gain negative. Tuning per mask makes the reported
    gain mean "best achievable ISTA for this mask vs zero-filling" instead of
    "one fixed threshold vs zero-filling".

    Selection uses validation images only — never the evaluation split.
    """
    ista_cfg = cfg["recon"]["wavelet_ista"]
    grid = [float(t) for t in ista_cfg.get("threshold_grid", DEFAULT_THRESHOLD_GRID)]
    noise_std = float(cfg.get("measurement", {}).get("noise_std", 0.0))
    y = recon.simulate_measurements(
        val_images, mask, noise_std=noise_std,
        generator=torch.Generator().manual_seed(int(cfg["seed"])),
    )

    rows = []
    for threshold in grid:
        estimate = recon.wavelet_ista(
            y, mask, threshold=threshold,
            n_iters=int(ista_cfg.get("n_iters", 50)),
            wavelet=str(ista_cfg.get("wavelet", "db4")),
            levels=int(ista_cfg.get("levels", 3)),
            final_dc=bool(ista_cfg.get("final_dc", True)),
        )
        psnr = float(
            np.mean([
                metrics.psnr(estimate[i].abs().numpy(), val_images[i].numpy())
                for i in range(val_images.shape[0])
            ])
        )
        rows.append({"threshold": threshold, "val_psnr": psnr})

    table = pd.DataFrame(rows)
    best = float(table.loc[table["val_psnr"].idxmax(), "threshold"])
    return best, table


def argumentation_table(
    mask_dict: dict[str, np.ndarray],
    frame: pd.DataFrame,
    train_power: np.ndarray,
    mass: dict[str, np.ndarray] | None = None,
    energies: dict[str, float] | None = None,
    basis: np.ndarray | None = None,
    baseline_method: str = "zero_filled",
    noise_std: float = 0.0,
) -> pd.DataFrame:
    """Design-time predictors next to measured outcomes, one row per mask.

    The point is argumentative rather than descriptive: every predictor claims
    to rank masks before any measurement is simulated, and the outcome columns
    test that claim. Per-method complex and magnitude MSE columns plus PSNR
    gains over the baseline are emitted for whichever methods are present in
    ``frame``.
    """
    rows = []
    for name, mask in mask_dict.items():
        sub = frame[frame["mask"] == name]
        base = sub[sub["method"] == baseline_method]
        n_samples = int(np.asarray(mask).sum())
        row: dict[str, Any] = {
            "mask": name,
            "actual_n_samples": n_samples,
            "actual_acceleration": float(mask.size / max(n_samples, 1)),
            "acquisition_family": acquisition_family(mask),
        }
        row["mask_score"] = artifacts.expected_zero_filled_mse(
            mask, train_power, noise_std=noise_std
        )
        row["prior_observable_energy_fraction"] = (
            artifacts.prior_observable_energy_fraction(mask, train_power)
        )
        row["prior_unobservable_energy_fraction"] = (
            1.0 - row["prior_observable_energy_fraction"]
        )
        row.update(artifacts.psf_metrics(mask))
        row.update(artifacts.spectrum_weighted_psf_metrics(mask, train_power))
        if mass is not None and energies is not None:
            row["wavelet_leakage"] = artifacts.wavelet_leakage_score(mask, mass, energies)
        if basis is not None:
            row["subspace_leakage"] = artifacts.subspace_nullspace_leakage(basis, mask)
        row["oracle_truth_nullspace_norm"] = float(
            base["oracle_truth_nullspace_norm"].mean()
        )
        for method in sorted(sub["method"].unique()):
            block = sub[sub["method"] == method]
            row[f"complex_mse_{method}"] = float(block["complex_mse"].mean())
            row[f"magnitude_mse_{method}"] = float(block["magnitude_mse"].mean())
            row[f"recon_nullspace_norm_{method}"] = float(
                block["recon_nullspace_norm"].mean()
            )
            row[f"measurement_residual_norm_{method}"] = float(
                block["measurement_residual_norm"].mean()
            )
            row[f"oracle_observed_subspace_error_norm_{method}"] = float(
                block["oracle_observed_subspace_error_norm"].mean()
            )
            if method != baseline_method:
                row[f"psnr_gain_{method}"] = float(
                    block["psnr"].mean() - base["psnr"].mean()
                )
        rows.append(row)
    return pd.DataFrame(rows)


def rank_correlations(
    table: pd.DataFrame, predictors: list[str], outcomes: list[str]
) -> pd.DataFrame:
    """Descriptive Spearman rank correlations over a constructed mask set.

    Masks in a design sweep are dependent, selected objects rather than iid
    observations, so this helper intentionally does not report classical
    significance p-values.
    """
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
            valid = table[[predictor, outcome]].dropna()
            if len(valid) < 2:
                continue
            rho, _ = spearmanr(valid[predictor], valid[outcome])
            rows.append(
                {
                    "predictor": predictor,
                    "outcome": outcome,
                    "n_masks": int(len(valid)),
                    "spearman_rho": float(rho),
                }
            )
    return pd.DataFrame(
        rows,
        columns=["predictor", "outcome", "n_masks", "spearman_rho"],
    )
