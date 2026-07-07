"""Greedy k-space sample selection strategies (NumPy)."""

from __future__ import annotations

import numpy as np

from .masks import center_indices, mask_from_indices, validate_budget


def _preselect_center(shape: tuple[int, int], n_center: int) -> np.ndarray:
    selected = np.zeros(shape[0] * shape[1], dtype=bool)
    selected[center_indices(shape, n_center)] = True
    return selected


def _aopt_gain(spectrum: np.ndarray, noise_var: float) -> np.ndarray:
    """Per-location reduction in expected posterior MSE.

    Under a diagonal Gaussian prior x_k ~ N(0, s_k) and noise variance
    sigma^2, the posterior MSE of the Wiener estimator after sampling set S is
        sum_{k not in S} s_k + sum_{k in S} s_k sigma^2 / (s_k + sigma^2),
    so adding location k reduces the expected MSE by s_k^2 / (s_k + sigma^2).
    """
    s = np.asarray(spectrum, dtype=np.float64).ravel()
    return s**2 / (s + noise_var)


def greedy_a_optimal(
    spectrum: np.ndarray,
    n_samples: int,
    noise_var: float = 1e-3,
    n_center: int = 0,
) -> np.ndarray:
    """Greedy Bayesian A-optimal selection with a diagonal k-space prior."""
    shape = spectrum.shape
    validate_budget(shape, n_samples, n_center)
    gain = _aopt_gain(spectrum, noise_var)
    selected = _preselect_center(shape, n_center)
    n_selected = int(selected.sum())
    while n_selected < n_samples:
        best = int(np.argmax(np.where(selected, -np.inf, gain)))
        selected[best] = True
        n_selected += 1
    return mask_from_indices(shape, np.flatnonzero(selected))


def _max_sidelobe(mask: np.ndarray) -> float:
    """Largest off-peak PSF magnitude relative to the peak (mask coherence)."""
    psf = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(mask), norm="ortho"))
    mag = np.abs(psf)
    cy, cx = mask.shape[0] // 2, mask.shape[1] // 2
    peak = mag[cy, cx]
    if peak == 0.0:
        return 1.0
    mag[cy, cx] = 0.0
    return float(mag.max() / peak)


def greedy_artifact_aware(
    spectrum: np.ndarray,
    n_samples: int,
    noise_var: float = 1e-3,
    beta: float = 0.5,
    n_candidates: int = 32,
    n_center: int = 0,
) -> np.ndarray:
    """Greedy selection trading expected MSE reduction against PSF coherence.

    Each step scores the top-`n_candidates` locations (ranked by A-optimal
    gain) with
        score(k | S) = gain(k) / max_gain - beta * max_sidelobe(S + {k})
    and adds the best one. The sidelobe term penalizes coherent sampling
    patterns whose PSF concentrates aliasing energy into replica peaks.
    """
    shape = spectrum.shape
    validate_budget(shape, n_samples, n_center)
    if n_candidates < 1:
        raise ValueError("n_candidates must be positive")
    gain = _aopt_gain(spectrum, noise_var)
    selected = _preselect_center(shape, max(n_center, 1))
    mask = mask_from_indices(shape, np.flatnonzero(selected))
    n_selected = int(selected.sum())

    while n_selected < n_samples:
        masked_gain = np.where(selected, -np.inf, gain)
        k = min(n_candidates, int((~selected).sum()))
        candidates = np.argpartition(masked_gain, -k)[-k:]
        candidates = candidates[np.isfinite(masked_gain[candidates])]
        max_gain = max(float(masked_gain[candidates].max()), 1e-12)

        best_idx, best_score = -1, -np.inf
        for cand in candidates:
            trial = mask.copy()
            trial.ravel()[cand] = 1.0
            score = float(masked_gain[cand]) / max_gain - beta * _max_sidelobe(trial)
            if score > best_score:
                best_idx, best_score = int(cand), score
        selected[best_idx] = True
        mask.ravel()[best_idx] = 1.0
        n_selected += 1
    return mask


def greedy_data_driven(
    images: np.ndarray,
    n_samples: int,
    n_center: int = 0,
) -> np.ndarray:
    """Greedy selection minimizing zero-filled error on training images.

    By Parseval, the zero-filled MSE summed over the training set equals the
    mean spectral energy at unsampled locations, so the exact greedy step adds
    the unsampled location with the largest mean |X_k|^2 measured on data.
    """
    images = np.asarray(images, dtype=np.float64)
    if images.ndim != 3:
        raise ValueError("images must have shape (n, H, W)")
    shape = images.shape[-2:]
    validate_budget(shape, n_samples, n_center)

    spectra = np.fft.fftshift(
        np.fft.fft2(np.fft.ifftshift(images, axes=(-2, -1)), norm="ortho"), axes=(-2, -1)
    )
    energy = (np.abs(spectra) ** 2).mean(axis=0).ravel()

    selected = _preselect_center(shape, n_center)
    n_selected = int(selected.sum())
    while n_selected < n_samples:
        best = int(np.argmax(np.where(selected, -np.inf, energy)))
        selected[best] = True
        n_selected += 1
    return mask_from_indices(shape, np.flatnonzero(selected))
