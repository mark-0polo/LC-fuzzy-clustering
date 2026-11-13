"""FCM: Bezdek fuzzy c-means with squared Euclidean distances."""
from time import perf_counter
import numpy as np
from ._constants import EPS, DEFAULT_MAX_ITER, DEFAULT_TOL, DEFAULT_RANDOM_STATE
from ._models import FitResult
from ._numerics import (safe_row_normalize, update_fcm_memberships_from_d2,
                        distances_to_centers_squared, fcm_centers, fcm_objective)


class FuzzyCMeans:
    def __init__(self, n_clusters, m=2.0, max_iter=DEFAULT_MAX_ITER, tol=DEFAULT_TOL, random_state=DEFAULT_RANDOM_STATE, init_centers=None):
        self.n_clusters = int(n_clusters)
        self.m = float(m)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.random_state = random_state
        self.init_centers = init_centers

    def fit(self, X):
        start = perf_counter()
        X = np.asarray(X, dtype=np.float64)
        n_samples = X.shape[0]

        if self.init_centers is None:
            rng = np.random.default_rng(self.random_state)
            U = rng.random((n_samples, self.n_clusters))
            U = safe_row_normalize(U)
        else:
            centers = np.asarray(self.init_centers, dtype=np.float64)
            U = update_fcm_memberships_from_d2(distances_to_centers_squared(X, centers), self.m)

        converged = False
        centers = None
        for iteration in range(1, self.max_iter + 1):
            U_old = U.copy()
            centers = fcm_centers(X, U, self.m)
            d2 = distances_to_centers_squared(X, centers)
            U = update_fcm_memberships_from_d2(d2, self.m)
            if np.linalg.norm(U - U_old, ord="fro") < self.tol:
                converged = True
                break

        objective = fcm_objective(X, U, centers, self.m)
        return FitResult(
            method="FCM",
            converged=converged,
            iterations=iteration,
            total_time_s=perf_counter() - start,
            objective=objective,
            U=U,
            centers=centers,
            extra={},
        )
