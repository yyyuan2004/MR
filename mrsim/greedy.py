"""Greedy frequency-domain measurement selection strategies.

Point-wise selectors use NumPy throughout; the reconstruction-in-the-loop
line selector additionally runs the torch-based iterative reconstruction.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.ndimage import binary_dilation

from . import recon
from .artifacts import subband_energies, subband_spectral_mass
from .masks import center_indices, fill_lines, mask_from_indices, radius_map, validate_budget
from .progress import track


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
    """Greedy Bayesian A-optimal selection with a diagonal frequency-domain prior."""
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


def _max_sidelobe(mask: np.ndarray, weight: np.ndarray | None = None) -> float:
    """Largest off-peak magnitude of the (optionally prior-weighted) PSF."""
    field = mask if weight is None else mask * weight
    psf = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(field), norm="ortho"))
    mag = np.abs(psf)
    cy, cx = mask.shape[0] // 2, mask.shape[1] // 2
    peak = mag[cy, cx]
    if peak == 0.0:
        return 1.0
    mag[cy, cx] = 0.0
    return float(mag.max() / peak)


def _sidelobe_reduction_candidates(
    mask: np.ndarray,
    selected: np.ndarray,
    weight: np.ndarray,
    n_wanted: int,
) -> np.ndarray:
    """Unselected locations whose addition best cancels the current worst sidelobe.

    The weighted PSF gains (weight_k / sqrt(N)) * exp(i * phase_k(tau)) when
    location k is added; picking k whose contribution opposes the PSF value at
    the worst off-peak lag tau* reduces that sidelobe the most.
    """
    n_rows, n_cols = mask.shape
    psf = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(mask * weight), norm="ortho"))
    mag = np.abs(psf)
    cy, cx = n_rows // 2, n_cols // 2
    mag[cy, cx] = 0.0
    ty, tx = np.unravel_index(int(np.argmax(mag)), mask.shape)
    u, v = ty - cy, tx - cx  # centered lag of the worst sidelobe

    ky = np.arange(n_rows) - cy
    kx = np.arange(n_cols) - cx
    a, b = np.meshgrid(ky, kx, indexing="ij")
    # Exact plane-wave value of the centered orthonormal inverse FFT of a
    # delta at centered frequency (a, b), evaluated at centered lag (u, v).
    phase = 2.0 * np.pi * (a * u / n_rows + b * v / n_cols) + np.pi * (a + b)
    contribution = (weight / np.sqrt(mask.size)) * np.exp(1j * phase)
    reduction = -np.real(np.conj(psf[ty, tx]) * contribution).ravel()
    reduction[selected] = -np.inf
    n_wanted = min(n_wanted, int((~selected).sum()))
    return np.argpartition(reduction, -n_wanted)[-n_wanted:]


def _hybrid_candidate_pool(
    gain: np.ndarray,
    selected: np.ndarray,
    mask: np.ndarray,
    weight: np.ndarray,
    n_candidates: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Union of four candidate families for the PSF-penalized greedy step.

    A pure top-gain pool clusters on the boundary of the low-frequency disk
    and makes the penalty ineffective; mixing in radius-weighted random draws,
    the boundary ring of the current support, and sidelobe-reduction
    candidates lets the penalty actually change the selection.
    """
    shape = mask.shape
    quota = max(1, n_candidates // 4)
    unselected = ~selected
    n_free = int(unselected.sum())
    masked_gain = np.where(selected, -np.inf, gain)

    pool: list[np.ndarray] = []
    k = min(quota, n_free)
    pool.append(np.argpartition(masked_gain, -k)[-k:])

    r = radius_map(shape).ravel()
    prob = np.where(unselected, (1.0 + r) ** (-2.0), 0.0)
    prob_sum = prob.sum()
    if prob_sum > 0:
        prob /= prob_sum
        pool.append(rng.choice(r.size, size=min(quota, n_free), replace=False, p=prob))

    ring = np.logical_and(binary_dilation(mask > 0.5), unselected.reshape(shape)).ravel()
    ring_idx = np.flatnonzero(ring)
    if ring_idx.size > 0:
        order = np.argsort(gain[ring_idx])[::-1]
        pool.append(ring_idx[order[:quota]])

    pool.append(_sidelobe_reduction_candidates(mask, selected, weight, quota))

    candidates = np.unique(np.concatenate(pool))
    return candidates[~selected[candidates]]


def greedy_psf_penalized_aopt(
    spectrum: np.ndarray,
    n_samples: int,
    noise_var: float = 1e-3,
    beta: float = 1.0,
    n_candidates: int = 32,
    n_center: int = 0,
    rng: np.random.Generator | None = None,
    weighted: bool = False,
    show_progress: bool = False,
) -> np.ndarray:
    """Greedy A-optimal selection penalized by the PSF max sidelobe.

    Each step scores a hybrid candidate pool with
        score(k | S) = gain(k) / max_gain - beta * sidelobe_norm(k | S)
    where sidelobe_norm is max_sidelobe(S + {k}) min-max normalized over the
    candidate pool. Normalization is essential: adding one location to an
    n-point mask changes the absolute sidelobe level only by O(1/n), so an
    unnormalized penalty is dominated by the O(1) gain gaps at every beta and
    the selection degenerates to plain A-opt. The prior enters through the
    gain term; the penalty measures arrangement coherence with the plain PSF
    (weighted=True switches to the prior-weighted PSF, which for steep priors
    is main-lobe dominated and discriminates poorly between candidates).
    """
    shape = spectrum.shape
    validate_budget(shape, n_samples, n_center)
    if n_candidates < 1:
        raise ValueError("n_candidates must be positive")
    rng = rng if rng is not None else np.random.default_rng(0)

    gain = _aopt_gain(spectrum, noise_var)
    weight = np.asarray(spectrum, dtype=np.float64) if weighted else np.ones(shape)
    weight = weight / weight.max()
    selected = _preselect_center(shape, max(n_center, 1))
    mask = mask_from_indices(shape, np.flatnonzero(selected))
    n_selected = int(selected.sum())

    steps: range | object = range(n_samples - n_selected)
    if show_progress:
        steps = track(steps, total=n_samples - n_selected, label="psf_penalized_aopt")
    for _ in steps:
        candidates = _hybrid_candidate_pool(gain, selected, mask, weight, n_candidates, rng)
        masked_gain = np.where(selected, -np.inf, gain)
        max_gain = max(float(masked_gain.max()), 1e-12)

        sidelobes = np.empty(candidates.size)
        for i, cand in enumerate(candidates):
            trial = mask.copy()
            trial.ravel()[cand] = 1.0
            sidelobes[i] = _max_sidelobe(trial, weight if weighted else None)
        spread = float(sidelobes.max() - sidelobes.min())
        sidelobe_norm = (sidelobes - sidelobes.min()) / spread if spread > 0 else np.zeros_like(sidelobes)

        scores = gain[candidates] / max_gain - beta * sidelobe_norm
        best_idx = int(candidates[int(np.argmax(scores))])
        selected[best_idx] = True
        mask.ravel()[best_idx] = 1.0
    return mask


def greedy_artifact_aware(
    spectrum: np.ndarray,
    n_samples: int,
    noise_var: float = 1e-3,
    beta: float = 0.5,
    n_candidates: int = 32,
    n_center: int = 0,
) -> np.ndarray:
    """Backward-compatibility alias for greedy_psf_penalized_aopt.

    Uses the plain (unweighted) PSF penalty and a deterministic candidate
    pool seed, matching the historical behavior as closely as possible.
    """
    return greedy_psf_penalized_aopt(
        spectrum,
        n_samples,
        noise_var=noise_var,
        beta=beta,
        n_candidates=n_candidates,
        n_center=n_center,
        rng=np.random.default_rng(0),
        weighted=False,
    )


def greedy_data_driven(
    images: np.ndarray,
    n_samples: int,
    n_center: int = 0,
) -> np.ndarray:
    """Greedy point selection minimizing zero-filled error on training images.

    By Parseval, the zero-filled MSE summed over the training set equals the
    mean spectral energy at unmeasured locations, so the exact greedy step
    adds the unmeasured location with the largest mean |X_k|^2.
    """
    energy = _mean_spectral_energy(images)
    shape = energy.shape
    validate_budget(shape, n_samples, n_center)

    flat = energy.ravel()
    selected = _preselect_center(shape, n_center)
    n_selected = int(selected.sum())
    while n_selected < n_samples:
        best = int(np.argmax(np.where(selected, -np.inf, flat)))
        selected[best] = True
        n_selected += 1
    return mask_from_indices(shape, np.flatnonzero(selected))


# ---------------------------------------------------------------------------
# Line-wise (Cartesian column) selection
# ---------------------------------------------------------------------------

def _mean_spectral_energy(images: np.ndarray) -> np.ndarray:
    images = np.asarray(images, dtype=np.float64)
    if images.ndim != 3:
        raise ValueError("images must have shape (n, H, W)")
    spectra = np.fft.fftshift(
        np.fft.fft2(np.fft.ifftshift(images, axes=(-2, -1)), norm="ortho"), axes=(-2, -1)
    )
    return (np.abs(spectra) ** 2).mean(axis=0)


def _lines_by_gain(
    column_gain: np.ndarray,
    shape: tuple[int, int],
    n_samples: int,
    n_center_lines: int,
) -> np.ndarray:
    """Column mask from a per-column gain: forced center lines, then greedy."""
    validate_budget(shape, n_samples)
    n_rows, n_cols = shape
    needed = int(np.ceil(n_samples / n_rows))
    n_center_lines = min(n_center_lines, needed)

    offsets = np.abs(np.arange(n_cols) - n_cols // 2)
    center_cols = np.argsort(offsets, kind="stable")[:n_center_lines].astype(np.int64)
    gain = np.asarray(column_gain, dtype=np.float64).copy()
    gain[center_cols] = -np.inf
    rest = np.argsort(gain, kind="stable")[::-1][: needed - n_center_lines]
    return fill_lines(shape, np.concatenate([center_cols, rest]), n_samples)


def greedy_line_a_optimal(
    spectrum: np.ndarray,
    n_samples: int,
    noise_var: float = 1e-3,
    n_center_lines: int = 2,
) -> np.ndarray:
    """A-optimal selection of whole Cartesian columns.

    The gain of a column is the sum of the per-location A-optimal gains it
    contains; with a diagonal prior, column gains are additive, so greedy
    column selection is exact.
    """
    gain = _aopt_gain(spectrum, noise_var).reshape(spectrum.shape)
    return _lines_by_gain(gain.sum(axis=0), spectrum.shape, n_samples, n_center_lines)


def greedy_lines_spectrum_energy(
    images: np.ndarray,
    n_samples: int,
    n_center_lines: int = 2,
) -> np.ndarray:
    """Column selection by empirical mean spectral energy of training data."""
    energy = _mean_spectral_energy(images)
    return _lines_by_gain(energy.sum(axis=0), energy.shape, n_samples, n_center_lines)


def greedy_lines_subspace_leakage(
    images: np.ndarray,
    n_samples: int,
    wavelet: str = "db4",
    levels: int = 3,
    n_center_lines: int = 2,
) -> np.ndarray:
    """Column selection minimizing wavelet-subspace leakage into the null space.

    Column gain = sum over subbands of (training energy in the subband) x
    (fraction of the subband's spectral mass on that column). Measuring the
    columns that the reconstruction basis needs most, this is an
    information-coverage criterion tied to the wavelet reconstruction rather
    than to raw Fourier energy.
    """
    images = np.asarray(images, dtype=np.float64)
    shape = images.shape[-2:]
    mass = subband_spectral_mass(shape, wavelet=wavelet, levels=levels)
    energies = subband_energies(images, wavelet=wavelet, levels=levels)
    column_gain = np.zeros(shape[1], dtype=np.float64)
    for key, band_mass in mass.items():
        column_gain += energies[key] * band_mass.sum(axis=0)
    return _lines_by_gain(column_gain, shape, n_samples, n_center_lines)


def greedy_lines_recon_in_loop(
    images: np.ndarray,
    n_samples: int,
    n_candidate_lines: int = 12,
    batch_size: int = 8,
    ista_threshold: float = 0.02,
    ista_iters: int = 6,
    wavelet: str = "db4",
    levels: int = 3,
    n_center_lines: int = 2,
    rng: np.random.Generator | None = None,
    show_progress: bool = False,
) -> np.ndarray:
    """Greedy column selection scored by an actual nonlinear reconstruction.

    Each step reconstructs a small training batch with a cheap wavelet-ISTA
    (few iterations, noiseless measurements) for every candidate column and
    keeps the column with the lowest reconstruction MSE. Unlike the spectral
    criteria, this accounts for what the nonlinear method can re-impute from
    the null space. Candidates mix top empirical-energy columns with random
    draws so the search is not confined to the low-frequency block.
    """
    images = np.asarray(images, dtype=np.float32)
    shape = images.shape[-2:]
    validate_budget(shape, n_samples)
    rng = rng if rng is not None else np.random.default_rng(0)

    n_rows, n_cols = shape
    needed = int(np.ceil(n_samples / n_rows))
    n_center_lines = min(n_center_lines, needed)
    offsets = np.abs(np.arange(n_cols) - n_cols // 2)
    chosen = list(np.argsort(offsets, kind="stable")[:n_center_lines].astype(int))

    column_energy = _mean_spectral_energy(images).sum(axis=0)
    batch = torch.from_numpy(images[:batch_size])

    def batch_mse(cols: list[int]) -> float:
        trial = np.zeros(shape, dtype=np.float32)
        trial[:, cols] = 1.0
        y = recon.simulate_measurements(batch, trial, noise_std=0.0)
        x = recon.wavelet_ista(
            y, trial,
            threshold=ista_threshold, n_iters=ista_iters,
            wavelet=wavelet, levels=levels, final_dc=True,
        )
        return float(((x.abs() - batch) ** 2).mean())

    steps: range | object = range(needed - len(chosen))
    if show_progress:
        steps = track(steps, total=needed - len(chosen), label="recon_in_loop")
    for _ in steps:
        free = np.setdiff1d(np.arange(n_cols), np.asarray(chosen, dtype=np.int64))
        half = max(1, n_candidate_lines // 2)
        by_energy = free[np.argsort(column_energy[free])[::-1][:half]]
        n_random = min(half, free.size)
        by_random = rng.choice(free, size=n_random, replace=False)
        candidates = np.unique(np.concatenate([by_energy, by_random]))

        best_col, best_mse = int(candidates[0]), np.inf
        for cand in candidates:
            mse = batch_mse(chosen + [int(cand)])
            if mse < best_mse:
                best_col, best_mse = int(cand), mse
        chosen.append(best_col)

    return fill_lines(shape, np.asarray(chosen, dtype=np.int64), n_samples)
