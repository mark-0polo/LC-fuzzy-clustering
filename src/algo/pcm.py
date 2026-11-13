"""PCM: fixed FCM-derived scales and independent typicalities."""
from time import perf_counter
import numpy as np
from ._constants import EPS, DEFAULT_MAX_ITER, DEFAULT_TOL, DEFAULT_RANDOM_STATE
from ._models import FitResult
from ._numerics import (safe_row_normalize, update_fcm_memberships_from_d2,
                        distances_to_centers_squared, fcm_centers, fcm_objective)
from .fcm import FuzzyCMeans


class PossibilisticCMeans:
    def __init__(self, n_clusters, m=2.0, max_iter=DEFAULT_MAX_ITER, tol=DEFAULT_TOL, random_state=DEFAULT_RANDOM_STATE):
        self.n_clusters = int(n_clusters)
        self.m = float(m)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.random_state = random_state

    @staticmethod
    def _eta_from_fcm(X, U, centers, m):
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
        centers = fcm.centers.copy()
        eta = self._eta_from_fcm(X, fcm.U, centers, self.m)
        d2 = distances_to_centers_squared(X, centers)
        T = 1.0 / (1.0 + (d2 / eta[None, :]) ** (1.0 / (self.m - 1.0)))

        converged = False
        for iteration in range(1, self.max_iter + 1):
            old_centers = centers.copy()
            d2 = distances_to_centers_squared(X, old_centers)
            T = 1.0 / (1.0 + (d2 / eta[None, :]) ** (1.0 / (self.m - 1.0)))
            weights = T ** self.m
            centers = (weights.T @ X) / np.maximum(weights.sum(axis=0)[:, None], EPS)
            center_change = np.linalg.norm(centers - old_centers, ord='fro') / centers.size
            if center_change <= self.tol:
                converged = True
                break

        d2 = distances_to_centers_squared(X, centers)
        objective = float(np.sum((T ** self.m) * d2) + np.sum(eta[None, :] * ((1.0 - T) ** self.m)))
        return FitResult(
            method="PCM",
            converged=converged,
            iterations=iteration,
            total_time_s=perf_counter() - start,
            objective=objective,
            U=T,
            centers=centers,
            extra={"eta_mean": float(np.mean(eta)), 'final_center_change': float(center_change),
                   'stopping': 'center_Frobenius_norm/(features*clusters)'},
            membership_kind='possibilistic', labels=np.argmax(T, axis=1), typicalities=T,
        )
