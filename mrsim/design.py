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


def diagonal_logdet(
    weights: np.ndarray, power_spectrum: np.ndarray, noise_var: float
) -> float:
    """D-optimal objective ``logdet(diag(w)/sigma^2 + Lambda^-1)`` up to a constant.

    ``weights`` may be fractional, which is what makes the design problem
    convex. A binary mask is the special case ``w in {0, 1}``.
    """
    w = np.asarray(weights, dtype=np.float64)
    power = np.asarray(power_spectrum, dtype=np.float64)
    if w.shape != power.shape:
        raise ValueError("weights and power_spectrum must have the same shape")
    if np.any(w < -1e-12) or np.any(w > 1.0 + 1e-12):
        raise ValueError("weights must lie in [0, 1]")
    if not np.isfinite(noise_var) or noise_var <= 0.0:
        raise ValueError("noise_var must be finite and positive")
    with np.errstate(divide="ignore"):
        prior_precision = np.where(power > 0.0, 1.0 / np.maximum(power, 1e-300), 0.0)
    active = power > 0.0
    return float(np.log(w[active] / noise_var + prior_precision[active]).sum())


def diagonal_relaxed_optimum(
    power_spectrum: np.ndarray,
    budget: int,
    noise_var: float,
    *,
    tol: float = 1e-12,
    max_iter: int = 200,
) -> tuple[np.ndarray, float]:
    """Exact optimum of the Joshi-Boyd relaxation for a diagonal prior.

    Maximizing ``sum_k log(w_k / sigma^2 + 1/s_k)`` over ``0 <= w <= 1`` with
    ``sum w = m`` is separable and concave, and the stationarity condition gives
    ``w_k = clip(t - sigma^2 / s_k, 0, 1)`` for a single scalar ``t`` fixed by
    the budget -- reverse water-filling. Bisection on ``t`` solves it to machine
    precision, so no solver dependency is needed.

    Because the relaxation's feasible set contains every binary mask of the same
    budget, its optimum is a genuine **upper bound** on what any mask can
    achieve. It is not a tight one: the optimal ``w`` is largely fractional, and
    for a diagonal prior the bound sits tens of nats above the true binary
    optimum. That difference is an integrality gap, a property of the
    relaxation, and reading it as suboptimality of a mask would be wrong -- see
    :func:`diagonal_binary_optimum`, which is exact here and is what
    :func:`optimality_gap` reports against.

    The solution does order locations by ``s_k``, so rounding it returns the
    top-k mask, which is the exact binary optimum. The relaxation therefore
    recovers the right design while overstating what is achievable.
    """
    power = np.asarray(power_spectrum, dtype=np.float64)
    if not np.isfinite(power).all() or np.any(power < 0.0):
        raise ValueError("power_spectrum must contain finite, non-negative values")
    if not np.isfinite(noise_var) or noise_var <= 0.0:
        raise ValueError("noise_var must be finite and positive")
    n_available = int((power > 0.0).sum())
    if not 1 <= budget <= max(n_available, 1):
        raise ValueError(
            f"budget={budget} must lie in [1, {n_available}] (locations with "
            "positive prior power)"
        )

    offset = np.where(power > 0.0, noise_var / np.maximum(power, 1e-300), np.inf)

    def allocation(t: float) -> np.ndarray:
        return np.clip(t - offset, 0.0, 1.0)

    low, high = 0.0, float(np.max(offset[np.isfinite(offset)]) + 1.0)
    for _ in range(max_iter):
        middle = 0.5 * (low + high)
        if allocation(middle).sum() < budget:
            low = middle
        else:
            high = middle
        if high - low < tol:
            break
    weights = allocation(0.5 * (low + high))
    return weights, diagonal_logdet(weights, power, noise_var)


def diagonal_binary_optimum(
    power_spectrum: np.ndarray, budget: int, noise_var: float
) -> float:
    """Exact best D-optimal objective over binary masks of a given budget.

    The objective separates as ``const + sum_{k in Omega} log(1 + s_k/sigma^2)``,
    and the per-location gain is monotone in ``s_k``, so the binary optimum is a
    top-k rule and is available in closed form -- no relaxation and no rounding
    needed. This is the same degeneracy :func:`diagonal_bayes` documents for the
    A-optimal criterion, now for the D-optimal one.
    """
    power = np.asarray(power_spectrum, dtype=np.float64)
    if not np.isfinite(power).all() or np.any(power < 0.0):
        raise ValueError("power_spectrum must contain finite, non-negative values")
    if not np.isfinite(noise_var) or noise_var <= 0.0:
        raise ValueError("noise_var must be finite and positive")
    if not 1 <= budget <= power.size:
        raise ValueError(f"budget={budget} must lie in [1, {power.size}]")
    baseline = diagonal_logdet(np.zeros_like(power), power, noise_var)
    gains = np.sort(np.log1p(power.ravel() / noise_var))[::-1]
    return float(baseline + gains[:budget].sum())


def optimality_gap(
    mask: np.ndarray, power_spectrum: np.ndarray, noise_var: float
) -> dict[str, float]:
    """How far a mask sits below the best achievable D-optimal objective.

    The reported gap is against :func:`diagonal_binary_optimum`, which is exact
    for this prior, so a gap of zero genuinely certifies the mask as optimal and
    a positive gap is exactly what a better design of the same budget would
    gain. The package could not previously make a statement of this kind at all;
    every comparison was relative to other heuristics.

    The convex relaxation's value is reported alongside for reference, but it
    is a loose upper bound here and its distance from the binary optimum is an
    integrality gap rather than anything about the mask.

    One caveat worth stating where it will be seen: the ``(1 - 1/e)`` greedy
    guarantee applies to this monotone submodular logdet objective, **not** to
    the A-optimal trace objective that ``greedy.greedy_a_optimal`` and
    ``greedy.greedy_subspace_aoptimal`` actually optimize. The trace objective
    is not submodular in general, so that guarantee must not be transferred to
    them.
    """
    mask_array, power = _validated_inputs(mask, power_spectrum, noise_var)
    budget = int(mask_array.sum())
    achieved = diagonal_logdet(mask_array, power, noise_var)
    best = diagonal_binary_optimum(power, budget, noise_var)
    _, relaxed = diagonal_relaxed_optimum(power, budget, noise_var)
    return {
        "achieved_logdet": achieved,
        "binary_optimum": best,
        "relaxed_upper_bound": relaxed,
        "optimality_gap_nats": float(best - achieved),
        "integrality_gap_nats": float(relaxed - best),
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
