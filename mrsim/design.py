"""Exact Gaussian design quantities for a binary frequency-domain mask.

The design diagnostics elsewhere in this package are proxies: ``rho`` and
``wavelet_leakage`` measure coverage, ``psf_max_sidelobe`` measures arrangement,
``mask_score`` predicts the error of one particular estimator. Under a Gaussian
prior and Gaussian noise none of them is necessary, because the two quantities
the design problem is actually about have closed forms:

- the Bayes MMSE, the error no estimator can beat under the stated prior, and
- the mutual information ``I(x; y)``, the information the design acquires.

Both are computed here exactly rather than approximated. Two consequences worth
stating plainly. First, any coverage metric of the form ``sum_k w_k M_k`` is a
*linear functional* of the mask, so two such metrics with similar weights are
rank-correlated by construction and cannot be independent axes -- the exact
quantities below are not linear in ``M`` and are the right replacement. Second,
for a prior that is diagonal in the measurement basis the optimal design is
exactly a top-k rule (see :func:`greedy_a_optimal`), so in that regime there is
no design problem to solve; the content lives in non-diagonal priors.

Conventions
-----------
``noise_var`` is the circular complex noise variance, matching
``recon.simulate_measurements`` where ``E|n_k|^2 = noise_std**2``. Mutual
information is returned in nats. For a circular complex Gaussian the mutual
information is ``logdet(I + A Sigma A^H / sigma^2)`` with *no* factor of one
half; the one-half belongs to the real-valued case.
"""

from __future__ import annotations

import numpy as np

from .artifacts import conjugate_partner_index, self_conjugate_positions


def _validated_inputs(
    mask: np.ndarray, power_spectrum: np.ndarray, noise_var: float
) -> tuple[np.ndarray, np.ndarray]:
    mask_array = np.asarray(mask, dtype=np.float64)
    power = np.asarray(power_spectrum, dtype=np.float64)
    if mask_array.shape != power.shape:
        raise ValueError(
            f"mask shape {mask_array.shape} does not match power spectrum shape "
            f"{power.shape}"
        )
    if not np.isin(mask_array, (0.0, 1.0)).all():
        raise ValueError("mask must be binary")
    if not np.isfinite(power).all() or np.any(power < 0.0):
        raise ValueError("power_spectrum must contain finite, non-negative values")
    if not np.isfinite(noise_var) or noise_var <= 0.0:
        raise ValueError("noise_var must be finite and positive")
    return mask_array, power


def diagonal_bayes(
    mask: np.ndarray,
    power_spectrum: np.ndarray,
    noise_var: float,
    *,
    hermitian: bool = True,
) -> dict[str, float]:
    """Exact Bayes MMSE and mutual information under a diagonal spectral prior.

    The prior is ``X_k ~ CN(0, s_k)`` independently across the frequency grid,
    which is the stationary-signal model the rest of the package already fits
    via ``experiment.mean_power_spectrum``. Because the Fourier transform is
    unitary, the posterior factorizes and everything is available in closed
    form.

    With ``hermitian`` (the default, correct for the real-valued signals this
    package generates), conjugate pairs are treated as the single complex
    unknown they are: measuring both members of an orbit yields two noisy looks
    at one value, which lowers the posterior variance through noise averaging
    but does not open a new degree of freedom. Setting ``hermitian=False``
    treats every grid location as an independent complex unknown, which
    overstates both the acquired information and the achievable accuracy for a
    real signal.

    Returns the total MMSE over the grid (``sum`` of posterior variances, i.e.
    ``E||x - x_hat||^2`` by Parseval), the same value per pixel, and the mutual
    information in nats.
    """
    mask_array, power = _validated_inputs(mask, power_spectrum, noise_var)

    if not hermitian:
        # Independent circular complex coefficients: posterior variance is the
        # prior variance where unmeasured and the Wiener shrinkage elsewhere.
        posterior = np.where(
            mask_array > 0.5, power * noise_var / (power + noise_var), power
        )
        information = float(
            np.log1p(np.where(mask_array > 0.5, power / noise_var, 0.0)).sum()
        )
        total = float(posterior.sum())
        return {
            "bayes_mmse_total": total,
            "bayes_mmse_per_pixel": total / mask_array.size,
            "mutual_information_nats": information,
        }

    rows, cols = conjugate_partner_index(mask_array.shape)
    reflected = mask_array[np.ix_(rows, cols)]
    # Looks at the orbit's single complex unknown, counting each measured member.
    n_looks = mask_array + reflected
    self_conjugate = self_conjugate_positions(mask_array.shape)
    # A self-conjugate frequency is its own partner, so the sum above
    # double-counts it, and its value is real rather than complex.
    n_looks = np.where(self_conjugate, mask_array, n_looks)

    with np.errstate(divide="ignore", invalid="ignore"):
        # Complex orbit: precision 1/s + n/sigma^2. Real (self-conjugate)
        # frequency: only the real part of the observation informs it, and that
        # part carries noise variance sigma^2/2, hence precision 1/s + 2n/sigma^2.
        looks_precision = np.where(self_conjugate, 2.0 * n_looks, n_looks)
        posterior = np.where(
            power > 0.0,
            power * noise_var / (noise_var + looks_precision * power),
            0.0,
        )
    # Each member of a size-two orbit carries the same posterior variance, so
    # summing over the whole grid already accounts for both.
    total = float(posterior.sum())

    # Information is per orbit, so count each size-two orbit once.
    orbit_weight = np.where(self_conjugate, 1.0, 0.5)
    per_orbit = np.where(
        self_conjugate,
        0.5 * np.log1p(2.0 * n_looks * power / noise_var),
        np.log1p(n_looks * power / noise_var),
    )
    information = float((orbit_weight * per_orbit).sum())

    return {
        "bayes_mmse_total": total,
        "bayes_mmse_per_pixel": total / mask_array.size,
        "mutual_information_nats": information,
    }


def subspace_bayes(
    mask: np.ndarray,
    basis: np.ndarray,
    eigenvalues: np.ndarray,
    noise_var: float,
) -> dict[str, float]:
    """Exact Bayes MMSE and mutual information under a Gaussian subspace prior.

    Model ``x = B alpha`` with ``alpha ~ CN(0, Lambda)`` and
    ``y_Omega = Phi_Omega alpha + eps``, ``Phi = F B``. The posterior precision
    is ``Lambda^-1 + Phi_Omega^H Phi_Omega / sigma^2``, so

        MMSE  = tr((Lambda^-1 + Phi_Omega^H Phi_Omega / sigma^2)^-1)
        I     = logdet(I + Lambda Phi_Omega^H Phi_Omega / sigma^2)

    and with orthonormal basis columns the coefficient-domain trace equals the
    image-domain one. This is the circular-complex convention, chosen to match
    ``recon.subspace_recon``, which solves the same complex normal equations;
    treating ``alpha`` as real instead would put a factor of two on the data
    term.

    Unlike :func:`diagonal_bayes` this prior is *not* diagonal in the
    measurement basis, which is exactly why the design problem stops being a
    top-k rule here.
    """
    from .subspace import to_kspace_basis

    mask_array = np.asarray(mask, dtype=np.float64)
    if not np.isin(mask_array, (0.0, 1.0)).all():
        raise ValueError("mask must be binary")
    if not np.isfinite(noise_var) or noise_var <= 0.0:
        raise ValueError("noise_var must be finite and positive")
    basis = np.asarray(basis)
    variances = np.asarray(eigenvalues, dtype=np.float64)
    if variances.shape != (basis.shape[1],):
        raise ValueError(f"eigenvalues must have shape ({basis.shape[1]},)")
    if not np.isfinite(variances).all() or np.any(variances < 0.0):
        raise ValueError("eigenvalues must contain finite, non-negative values")
    variances = np.maximum(variances, 1e-12)

    omega = np.flatnonzero(mask_array.ravel() > 0.5)
    phi_omega = to_kspace_basis(basis, mask_array.shape)[omega]
    gram = phi_omega.conj().T @ phi_omega / noise_var

    precision = gram + np.diag(1.0 / variances)
    posterior = np.linalg.inv(precision)
    total = float(np.trace(posterior).real)

    sign, logabsdet = np.linalg.slogdet(
        np.eye(variances.size) + np.diag(variances) @ gram
    )
    if sign.real <= 0.0:
        raise FloatingPointError("non-positive determinant in the information term")

    return {
        "bayes_mmse_total": total,
        "bayes_mmse_per_pixel": total / mask_array.size,
        "mutual_information_nats": float(logabsdet),
    }
