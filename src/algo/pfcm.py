"""PFCM: joint fuzzy memberships and possibilistic typicalities."""
from time import perf_counter
import numpy as np
from ._constants import EPS, DEFAULT_MAX_ITER, DEFAULT_TOL, DEFAULT_RANDOM_STATE
from ._models import FitResult
from ._numerics import (safe_row_normalize, update_fcm_memberships_from_d2,
                        distances_to_centers_squared, fcm_centers, fcm_objective)
from .fcm import FuzzyCMeans


class PossibilisticFuzzyCMeans:
    def __init__(self, n_clusters, m=2.0, nu=2.0, a=1.0, b=1.0, max_iter=DEFAULT_MAX_ITER, tol=DEFAULT_TOL, random_state=DEFAULT_RANDOM_STATE):
        self.n_clusters = int(n_clusters)
        self.m = float(m)
        self.nu = float(nu)
        self.a = float(a)
        self.b = float(b)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.random_state = random_state

    @staticmethod
    def _eta(X, U, centers, m):
        d2 = distances_to_centers_squared(X, centers)
        weights = U ** m
        return np.maximum(np.sum(weights * d2, axis=0) / np.maximum(weights.sum(axis=0), EPS), EPS)

    def fit(self, X):
        start = perf_counter()
        X = np.asarray(X, dtype=np.float64)

        fcm = FuzzyCMeans(
            self.n_clusters,
            m=self.m,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state,
        ).fit(X)
        U = fcm.U.copy()
        centers = fcm.centers.copy()
        eta = self._eta(X, U, centers, self.m)
        d2 = distances_to_centers_squared(X, centers)
        T = 1.0 / (1.0 + ((self.b * d2) / eta[None, :]) ** (1.0 / (self.nu - 1.0)))

        converged = False
        for iteration in range(1, self.max_iter + 1):
            old_centers = centers.copy()
            d2 = distances_to_centers_squared(X, old_centers)
            U = update_fcm_memberships_from_d2(d2, self.m)
            T = 1.0 / (1.0 + ((self.b * d2) / eta[None, :]) ** (1.0 / (self.nu - 1.0)))
            weights = self.a * (U ** self.m) + self.b * (T ** self.nu)
            centers = (weights.T @ X) / np.maximum(weights.sum(axis=0)[:, None], EPS)
            delta = np.linalg.norm(centers - old_centers, ord='fro') / centers.size
            if delta <= self.tol:
                converged = True
                break

        d2 = distances_to_centers_squared(X, centers)
        objective = float(
            self.a * np.sum((U ** self.m) * d2)
            + self.b * np.sum((T ** self.nu) * d2)
            + np.sum(eta[None, :] * ((1.0 - T) ** self.nu))
        )
        return FitResult(
            method="PFCM",
            converged=converged,
            iterations=iteration,
            total_time_s=perf_counter() - start,
            objective=objective,
            U=U,
            centers=centers,
            extra={"eta_mean": float(np.mean(eta)), 'final_center_change': float(delta),
                   'stopping': 'center_Frobenius_norm/(features*clusters)'},
            typicalities=T,
        )
