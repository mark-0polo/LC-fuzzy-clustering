"""eFCM: entropy-regularized fuzzy clustering at fixed temperature."""
from time import perf_counter
import numpy as np
from ._constants import EPS, DEFAULT_MAX_ITER, DEFAULT_TOL, DEFAULT_RANDOM_STATE
from ._models import FitResult
from ._numerics import (safe_row_normalize, update_fcm_memberships_from_d2,
                        distances_to_centers_squared, fcm_centers, fcm_objective)


class EntropyRegularizedFCM:
    def __init__(self, n_clusters, max_iter=DEFAULT_MAX_ITER, tol=DEFAULT_TOL, random_state=DEFAULT_RANDOM_STATE, lam=None):
        self.n_clusters = int(n_clusters)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.random_state = random_state
        self.lam = lam

    @staticmethod
    def _soft_memberships(d2, lam):
        scores = -d2 / max(lam, EPS)
        scores -= np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(scores)
        return exp_scores / np.maximum(exp_scores.sum(axis=1, keepdims=True), EPS)

    def fit(self, X):
        start = perf_counter()
        X = np.asarray(X, dtype=np.float64)
        rng = np.random.default_rng(self.random_state)
        U = rng.random((X.shape[0], self.n_clusters))
        U = safe_row_normalize(U)

        # Paper equation (4): exp(-lambda_paper*d2). Here lam is its reciprocal.
        lam = 1.0 if self.lam is None else float(self.lam)
        if not np.isfinite(lam) or lam <= 0:
            raise ValueError('Entropy temperature must be finite and positive.')

        converged = False
        centers = None
        for iteration in range(1, self.max_iter + 1):
            U_old = U.copy()
            denominator = np.maximum(U.sum(axis=0)[:, None], EPS)
            centers = (U.T @ X) / denominator
            d2 = distances_to_centers_squared(X, centers)
            U = self._soft_memberships(d2, lam)
            if np.linalg.norm(U - U_old, ord="fro") < self.tol:
                converged = True
                break

        objective = float(np.sum(U * distances_to_centers_squared(X, centers)) + lam * np.sum(U * np.log(np.maximum(U, EPS))))
        return FitResult(
            method="eFCM",
            converged=converged,
            iterations=iteration,
            total_time_s=perf_counter() - start,
            objective=objective,
            U=U,
            centers=centers,
            extra={'temperature': lam, 'lambda_paper': 1.0 / lam},
        )
