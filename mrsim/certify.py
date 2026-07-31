"""Split-conformal risk certificates for a fixed mask and reconstruction.

``docs/experiments.md`` lists a per-sample error certificate as unfinished
theory. Half of that question is settled and the other half is achievable, and
it is worth separating them cleanly.

**What is impossible.** Without a prior, the set of signals consistent with a
measurement is ``{x : ||A x - y|| <= eps}``. Along the null space of ``A`` that
set is unbounded, so ``sup ||(I - P)(x_hat - x)|| = inf`` over it: no
assumption-free bound on the null-space error exists, at any confidence level.
This needs no theorem, only the observation that the feasible set has infinite
diameter in those directions.

**What is also impossible, but less obviously.** Even granting a distribution,
*conditional* coverage -- an interval valid given the particular measurement
observed -- cannot be attained distribution-free with finite samples for
non-atomic distributions (Vovk; Barber, Candes, Ramdas and Tibshirani, "The
limits of distribution-free conditional predictive inference"). Nothing here
claims it.

**What is achievable, and is implemented here.** Marginal coverage under
exchangeability. Calibrate the reconstruction error on a held-out split, take
an empirical quantile with the finite-sample correction, and the resulting
bound covers a fresh sample with probability at least ``1 - alpha``. The
guarantee is over the draw of calibration and test points jointly; it is not a
statement about any one image, and it transfers to a shifted distribution only
insofar as exchangeability survives the shift -- which for the prior-shift
experiments this package contemplates, it does not.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ConformalCertificate:
    """Calibrated error bound plus the facts needed to interpret it."""

    alpha: float
    threshold: float
    n_calibration: int
    score_name: str

    @property
    def nominal_coverage(self) -> float:
        return 1.0 - self.alpha

    def covers(self, scores: np.ndarray) -> np.ndarray:
        """Elementwise indicator that the realized score is within the bound."""
        return np.asarray(scores, dtype=np.float64) <= self.threshold

    def empirical_coverage(self, scores: np.ndarray) -> float:
        return float(np.mean(self.covers(scores)))


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample conformal quantile of calibration scores.

    Uses the ``ceil((n + 1) (1 - alpha)) / n`` empirical quantile, the standard
    correction that makes marginal coverage hold at level ``1 - alpha`` for a
    fresh exchangeable draw rather than only asymptotically. When the required
    rank exceeds ``n`` the bound is infinite: with that few calibration points
    no finite threshold is justified, and returning ``inf`` states that instead
    of quietly returning the maximum observed score.
    """
    values = np.asarray(scores, dtype=np.float64).ravel()
    if values.size == 0:
        raise ValueError("need at least one calibration score")
    if not np.isfinite(values).all():
        raise ValueError("calibration scores must be finite")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between 0 and 1")
    n = values.size
    rank = math.ceil((n + 1) * (1.0 - alpha))
    if rank > n:
        return float("inf")
    return float(np.sort(values)[rank - 1])


def calibrate(
    calibration_scores: np.ndarray,
    alpha: float = 0.1,
    *,
    score_name: str = "reconstruction_error_norm",
) -> ConformalCertificate:
    """Build a certificate from held-out reconstruction errors."""
    threshold = conformal_quantile(calibration_scores, alpha)
    return ConformalCertificate(
        alpha=float(alpha),
        threshold=threshold,
        n_calibration=int(np.asarray(calibration_scores).size),
        score_name=score_name,
    )


def coverage_interval(n: int, alpha: float, confidence: float = 0.95) -> tuple[float, float]:
    """Normal-approximation interval for an observed coverage rate.

    Reported next to the empirical coverage so a small test split is not read
    as evidence of under- or over-coverage. With the handful of test images the
    default configuration ships, this interval is wide enough that almost
    nothing is distinguishable, which is itself the point.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    from scipy.stats import norm

    p = 1.0 - alpha
    half = float(norm.ppf(0.5 + confidence / 2.0)) * math.sqrt(p * (1.0 - p) / n)
    return max(0.0, p - half), min(1.0, p + half)


def error_scores(reconstructions: np.ndarray, truths: np.ndarray) -> np.ndarray:
    """Per-image Euclidean reconstruction error, the default conformal score."""
    recon = np.asarray(reconstructions)
    truth = np.asarray(truths)
    if recon.shape != truth.shape:
        raise ValueError(
            f"reconstruction shape {recon.shape} does not match truth shape {truth.shape}"
        )
    difference = np.abs(recon - truth).reshape(recon.shape[0], -1)
    return np.linalg.norm(difference, axis=1)
