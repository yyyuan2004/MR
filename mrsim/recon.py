"""Measurement simulation and simple reconstruction methods."""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
import pywt
import torch

from .fft_ops import as_mask_tensor, fft2c, ifft2c, projector


def sample_complex_noise_like(
    reference: torch.Tensor,
    noise_std: float,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Draw full-grid circular complex Gaussian noise like ``reference``.

    ``noise_std`` is the complex standard deviation, so every entry satisfies
    ``E[|n|^2] = noise_std^2``.  The returned tensor has the same shape and
    device as ``reference`` and uses complex64 or complex128 according to the
    reference precision.  Generate this tensor once and pass it explicitly to
    :func:`simulate_measurements` for every mask in a paired comparison.
    """
    if noise_std < 0.0:
        raise ValueError("noise_std must be non-negative")
    complex_dtype = (
        torch.complex128
        if reference.dtype in (torch.float64, torch.complex128)
        else torch.complex64
    )
    if noise_std == 0.0:
        return torch.zeros(
            reference.shape,
            dtype=complex_dtype,
            device=reference.device,
        )
    real_dtype = torch.float64 if complex_dtype == torch.complex128 else torch.float32
    sample_device = (
        reference.device
        if generator is None
        else torch.device(generator.device)
    )
    real = torch.randn(
        reference.shape,
        dtype=real_dtype,
        device=sample_device,
        generator=generator,
    )
    imag = torch.randn(
        reference.shape,
        dtype=real_dtype,
        device=sample_device,
        generator=generator,
    )
    noise = (noise_std / math.sqrt(2.0)) * torch.complex(real, imag)
    return noise.to(reference.device)


def simulate_measurements(
    images: torch.Tensor,
    mask: np.ndarray | torch.Tensor,
    noise_std: float = 0.0,
    generator: torch.Generator | None = None,
    *,
    noise: np.ndarray | torch.Tensor | None = None,
) -> torch.Tensor:
    """Simulate full-grid encoded measurements ``y = M (F x + n)``.

    By default, iid circular complex noise is sampled from ``noise_std`` and
    ``generator``.  Alternatively, ``noise`` supplies an already-scaled,
    full-k-space complex realization with exactly the same shape as ``images``.
    The same explicit tensor can therefore be reused across masks for paired
    comparisons.  Explicit ``noise`` is mutually exclusive with both
    ``noise_std`` and ``generator``.
    """
    if noise_std < 0.0:
        raise ValueError("noise_std must be non-negative")
    complex_dtype = (
        torch.complex128
        if images.dtype in (torch.float64, torch.complex128)
        else torch.complex64
    )
    k = fft2c(images.to(complex_dtype))
    if noise is not None:
        if noise_std != 0.0 or generator is not None:
            raise ValueError("explicit noise is mutually exclusive with noise_std and generator")
        noise_t = (
            torch.from_numpy(np.ascontiguousarray(noise))
            if isinstance(noise, np.ndarray)
            else noise
        )
        if not noise_t.is_complex():
            raise TypeError("explicit noise must be a complex tensor or array")
        if noise_t.shape != k.shape:
            raise ValueError(
                f"explicit noise shape {tuple(noise_t.shape)} does not match "
                f"full k-space shape {tuple(k.shape)}"
            )
        k = k + noise_t.to(device=k.device, dtype=k.dtype)
    elif noise_std > 0.0:
        k = k + sample_complex_noise_like(k, noise_std, generator=generator)
    return as_mask_tensor(mask, like=k) * k


def zero_filled(y: torch.Tensor) -> torch.Tensor:
    """Zero-filled reconstruction F^H y of already-masked frequency-domain data."""
    return ifft2c(y)


def ridge(
    y: torch.Tensor,
    mask: np.ndarray | torch.Tensor,
    lam: float,
    spectrum: np.ndarray | torch.Tensor | None = None,
    prior_mean: np.ndarray | torch.Tensor | None = None,
) -> torch.Tensor:
    """Ridge / diagonal-Gaussian posterior mean in the frequency domain.

    argmin_x ||M F x - y||^2 + lam ||x||^2 decouples in the frequency domain
    because F is unitary and M is diagonal: measured coefficients shrink by
    1/(1 + lam) and unmeasured coefficients are zero. With a diagonal prior
    power spectrum s_k, the shrinkage becomes the Wiener weight
    s_k / (s_k + lam).

    ``prior_mean`` optionally supplies the full complex frequency-domain prior
    mean ``mu``.  The posterior mean is then
    ``k_hat = mu + M * weight * (y - mu)``; unmeasured coefficients equal
    ``mu`` exactly.  Omitting it preserves the historical zero-mean result
    ``k_hat = M * weight * y``.
    """
    if not math.isfinite(lam) or lam < 0.0:
        raise ValueError("lam must be finite and non-negative")
    complex_dtype = (
        torch.complex128
        if y.dtype in (torch.float64, torch.complex128)
        else torch.complex64
    )
    y_c = y.to(complex_dtype)
    mask_t = as_mask_tensor(mask, like=y_c)
    if spectrum is None:
        weight: torch.Tensor | float = 1.0 / (1.0 + lam)
    else:
        s = (
            torch.as_tensor(
                np.asarray(spectrum),
                dtype=y_c.real.dtype,
                device=y_c.device,
            )
            if isinstance(spectrum, np.ndarray)
            else spectrum.to(device=y_c.device, dtype=y_c.real.dtype)
        )
        if not torch.isfinite(s).all() or torch.any(s < 0):
            raise ValueError("spectrum must contain finite, non-negative values")
        try:
            spectrum_shape = torch.broadcast_shapes(s.shape, y_c.shape)
        except RuntimeError as exc:
            raise ValueError(
                f"spectrum shape {tuple(s.shape)} is not broadcastable to "
                f"measurement shape {tuple(y_c.shape)}"
            ) from exc
        if spectrum_shape != y_c.shape:
            raise ValueError(
                f"spectrum shape {tuple(s.shape)} would expand the measurement "
                f"shape {tuple(y_c.shape)} to {tuple(spectrum_shape)}"
            )
        if lam == 0.0:
            weight = torch.where(s > 0.0, torch.ones_like(s), torch.zeros_like(s))
        else:
            weight = s / (s + lam)
    if prior_mean is None:
        k_hat = mask_t * weight * y_c
    else:
        mu = (
            torch.from_numpy(np.ascontiguousarray(prior_mean))
            if isinstance(prior_mean, np.ndarray)
            else prior_mean
        )
        mu = mu.to(device=y_c.device, dtype=y_c.dtype)
        try:
            broadcast_shape = torch.broadcast_shapes(mu.shape, y_c.shape)
        except RuntimeError as exc:
            raise ValueError(
                f"prior_mean shape {tuple(mu.shape)} is not broadcastable to "
                f"measurement shape {tuple(y_c.shape)}"
            ) from exc
        if broadcast_shape != y_c.shape:
            raise ValueError(
                f"prior_mean shape {tuple(mu.shape)} would expand the measurement "
                f"shape {tuple(y_c.shape)} to {tuple(broadcast_shape)}"
            )
        k_hat = mu + mask_t * weight * (y_c - mu)
    return ifft2c(k_hat)


def subspace_recon(
    y: torch.Tensor,
    mask: np.ndarray | torch.Tensor,
    basis: np.ndarray,
    lam: float = 1e-6,
    prior_mean: np.ndarray | torch.Tensor | None = None,
    coefficient_variances: np.ndarray | torch.Tensor | None = None,
) -> torch.Tensor:
    """Closed-form reconstruction under a subspace/manifold prior.

    Solves the regularized normal equations in the PCA coordinates, with
    ``Phi = F B``. When ``prior_mean`` is supplied, measurements are centered
    around ``A mean`` and the result is ``mean + B alpha_hat``. With PCA
    coefficient variances ``Lambda``, the system uses the Gaussian-prior
    precision ``lam * Lambda^-1``. Omitting them preserves isotropic
    regularization. ``lam`` should match the measurement noise variance.

    The estimate lies in an affine prior model, not in the observed subspace,
    so it can have nonzero null-space content. Whether that imputation is
    faithful is exactly what ``decompose_error`` measures offline.
    """
    from .subspace import to_kspace_basis

    if not math.isfinite(lam) or lam <= 0.0:
        raise ValueError("lam must be finite and positive")
    mask_np = (
        mask.detach().cpu().numpy()
        if isinstance(mask, torch.Tensor)
        else np.asarray(mask)
    )
    shape = mask_np.shape
    omega = np.flatnonzero(mask_np.ravel() > 0.5)
    basis = np.asarray(basis)
    phi_omega = to_kspace_basis(basis, shape)[omega]

    single = y.ndim == 2
    batch = y[None] if single else y
    y_array = (
        batch.detach().cpu().numpy().astype(np.complex128).reshape(batch.shape[0], -1)
    )
    mean_flat = np.zeros(basis.shape[0], dtype=np.complex128)
    if prior_mean is not None:
        mean_array = (
            prior_mean.detach().cpu().numpy()
            if isinstance(prior_mean, torch.Tensor)
            else np.asarray(prior_mean)
        )
        if mean_array.size != basis.shape[0]:
            raise ValueError(
                f"prior_mean contains {mean_array.size} entries, expected {basis.shape[0]}"
            )
        mean_flat = mean_array.reshape(-1).astype(np.complex128)
        mean_k = fft2c(
            torch.from_numpy(mean_flat.reshape(shape))
        ).numpy().reshape(-1)
        y_array = y_array - mask_np.reshape(1, -1) * mean_k.reshape(1, -1)
    y_omega = y_array[:, omega]

    if coefficient_variances is None:
        variances = np.ones(basis.shape[1], dtype=np.float64)
    else:
        variances = (
            coefficient_variances.detach().cpu().numpy()
            if isinstance(coefficient_variances, torch.Tensor)
            else np.asarray(coefficient_variances)
        )
        variances = np.asarray(variances, dtype=np.float64)
        if variances.shape != (basis.shape[1],):
            raise ValueError(
                f"coefficient_variances must have shape ({basis.shape[1]},)"
            )
        if not np.isfinite(variances).all() or np.any(variances < 0.0):
            raise ValueError(
                "coefficient_variances must contain finite, non-negative values"
            )
        variances = np.maximum(variances, 1e-12)
    gram = (
        phi_omega.conj().T @ phi_omega
        + lam * np.diag(1.0 / variances)
    )
    alpha = np.linalg.solve(gram, phi_omega.conj().T @ y_omega.T)
    x = (
        mean_flat[:, None] + basis.astype(np.complex128) @ alpha
    ).T.reshape(batch.shape)
    output_dtype = np.complex128 if y.dtype == torch.complex128 else np.complex64
    out = torch.from_numpy(x.astype(output_dtype)).to(y.device)
    return out[0] if single else out


def generative_recon(
    y: torch.Tensor,
    mask: np.ndarray | torch.Tensor,
    generator,
    z_init: torch.Tensor,
    steps: int = 200,
    lr: float = 0.05,
) -> torch.Tensor:
    """Latent-space reconstruction min_z ||M F G(z) - y||^2 by gradient descent.

    The generator maps a latent vector to an image; optimization runs in the
    latent space with Adam from z_init (single-image contract, deterministic
    given z_init). Like subspace_recon, the output lives on the generator
    manifold rather than in the observed subspace, so it imputes null-space
    content.
    """
    y = y.to(torch.complex64)
    mask_t = as_mask_tensor(mask, like=y)
    z = z_init.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([z], lr=lr)
    for _ in range(steps):
        optimizer.zero_grad()
        residual = mask_t * fft2c(generator(z).to(torch.complex64)) - y
        loss = (residual.abs() ** 2).sum()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        return generator(z).detach().to(torch.complex64)


class IstaResult(NamedTuple):
    """Wavelet-ISTA output with the per-iteration objective values."""

    image: torch.Tensor
    objective_history: list[float]


def _wavelet_decompose(x: np.ndarray, wavelet: str, levels: int) -> list:
    return pywt.wavedec2(x, wavelet=wavelet, mode="periodization", level=levels, axes=(-2, -1))


def _wavelet_soft_threshold(
    x: torch.Tensor, threshold: float, wavelet: str, levels: int
) -> torch.Tensor:
    """Complex-magnitude soft-thresholding in the wavelet domain.

    Shrinks each complex coefficient by max(1 - t/|c|, 0), which preserves the
    phase; real and imaginary parts share one decomposition (PyWavelets
    transforms complex input componentwise).
    """
    arr = x.detach().cpu().numpy().astype(np.complex128)
    coeffs = _wavelet_decompose(arr, wavelet, levels)

    def shrink(c: np.ndarray) -> np.ndarray:
        mag = np.abs(c)
        return c * np.maximum(1.0 - threshold / np.maximum(mag, 1e-12), 0.0)

    shrunk = [shrink(coeffs[0])] + [tuple(shrink(d) for d in band) for band in coeffs[1:]]
    out = pywt.waverec2(shrunk, wavelet=wavelet, mode="periodization", axes=(-2, -1))
    return torch.from_numpy(out.astype(np.complex64)).to(x.device)


def _wavelet_l1(x: torch.Tensor, wavelet: str, levels: int) -> float:
    coeffs = _wavelet_decompose(
        x.detach().cpu().numpy().astype(np.complex128), wavelet, levels
    )
    total = float(np.abs(coeffs[0]).sum())
    for band in coeffs[1:]:
        total += float(sum(np.abs(d).sum() for d in band))
    return total


def _ista_objective(
    x: torch.Tensor,
    y: torch.Tensor,
    mask_t: torch.Tensor,
    threshold: float,
    wavelet: str,
    levels: int,
) -> float:
    """Value of 0.5 * ||M F x - y||^2 + threshold * ||W x||_1."""
    residual = mask_t * fft2c(x) - y
    return 0.5 * float((residual.abs() ** 2).sum()) + threshold * _wavelet_l1(
        x, wavelet, levels
    )


def wavelet_ista(
    y: torch.Tensor,
    mask: np.ndarray | torch.Tensor,
    *,
    threshold: float,
    n_iters: int = 50,
    wavelet: str = "db4",
    levels: int = 3,
    final_dc: bool = True,
    return_history: bool = False,
    momentum: bool = True,
    tol: float = 0.0,
) -> torch.Tensor | IstaResult:
    """Solve min_x 0.5 * ||M F x - y||^2 + threshold * ||W x||_1.

    Step size 1.0 is valid because the forward operator satisfies A^H A = P
    (an orthogonal projector), so the data-fidelity gradient is 1-Lipschitz.
    Thresholding in the wavelet basis couples frequency coefficients, so
    iterates leave the observed subspace and impute null-space content, unlike
    zero filling or a zero-mean diagonal reconstruction.

    With ``momentum`` (the default) the solver is monotone FISTA: the proximal
    step is taken at an extrapolated point for the O(1/k^2) rate, and the
    iterate is only accepted when it does not increase the objective. Plain
    ISTA converges at O(1/k), which at the iteration counts used here leaves
    the solver far from the minimizer — and an unconverged solver confounds any
    comparison *between masks*, because the measured difference then mixes mask
    quality with solver dynamics. Pass ``momentum=False`` for the historical
    plain-ISTA iteration.

    ``tol`` optionally stops early once the relative objective decrease falls
    below it; the default 0.0 always runs the full ``n_iters``.

    If final_dc, one data-consistency step replaces measured frequency
    coefficients with the measurements. Deterministic: no randomness anywhere.
    With return_history, returns IstaResult(image, objective_history); the
    history holds the accepted objective after each proximal step and is
    non-increasing up to floating-point round-off in both modes.
    """
    if threshold < 0.0:
        raise ValueError("threshold must be non-negative")
    if tol < 0.0:
        raise ValueError("tol must be non-negative")
    y = y.to(torch.complex64)
    mask_t = as_mask_tensor(mask, like=y)
    adjoint = ifft2c(y)  # A^H y; y is already masked
    x = adjoint.clone()
    extrapolated = x.clone()
    t = 1.0
    history: list[float] = []
    # The monotone guard and the stopping rule both need the objective; plain
    # ISTA without history or tolerance keeps the cheaper transform-only loop.
    track_objective = return_history or momentum or tol > 0.0
    objective = (
        _ista_objective(x, y, mask_t, threshold, wavelet, levels)
        if track_objective
        else math.inf
    )

    for _ in range(n_iters):
        # Gradient of the data term is A^H (A z - y) = P z - A^H y.
        gradient_step = extrapolated - (projector(extrapolated, mask_t) - adjoint)
        candidate = _wavelet_soft_threshold(gradient_step, threshold, wavelet, levels)

        if not track_objective:
            x = candidate
            extrapolated = candidate
            continue

        candidate_objective = _ista_objective(
            candidate, y, mask_t, threshold, wavelet, levels
        )
        previous, previous_objective = x, objective
        if candidate_objective <= previous_objective:
            x, objective = candidate, candidate_objective
        history.append(objective)

        if momentum:
            # MFISTA: extrapolate from the accepted iterate through the
            # candidate, which keeps the accelerated rate while the guard above
            # keeps the objective sequence monotone.
            t_next = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t * t))
            extrapolated = (
                x
                + (t / t_next) * (candidate - x)
                + ((t - 1.0) / t_next) * (x - previous)
            )
            t = t_next
        else:
            extrapolated = x

        if tol > 0.0 and previous_objective - objective <= tol * max(
            abs(objective), 1e-12
        ):
            break

    if final_dc:
        x = x + ifft2c(y - mask_t * fft2c(x))
    if return_history:
        return IstaResult(image=x, objective_history=history)
    return x
