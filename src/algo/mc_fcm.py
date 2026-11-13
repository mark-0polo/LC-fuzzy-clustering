"""MC-FCM paper equations and procedure; independent paper implementation.

Bounds and power are study inputs, not alleged author defaults. Literal Section
3.1 includes self among floor(N/K) sorted distances; an explicit option permits
sensitivity to the prose's ambiguous 'surrounding' convention. Original-author
C# software is unavailable; this is an independently validated implementation.
"""
from time import perf_counter
import numpy as np
from ._constants import EPS, DEFAULT_MAX_ITER, DEFAULT_TOL, DEFAULT_RANDOM_STATE
from ._models import FitResult
from ._numerics import (pairwise_distances, safe_row_normalize, distances_to_centers_squared,
                       update_fcm_memberships_from_d2)


def neighbor_concentration(X, n_clusters, include_self=True):
    """Literal paper neighbor sum with a finite equal-density extension."""
    X = np.asarray(X, dtype=np.float64)
    n = len(X)
    if n < 2 or not 1 <= n_clusters <= n:
        raise ValueError('MC-FCM requires N>=2 and 1<=K<=N.')
    count = min(n if include_self else n - 1, max(1, n // n_clusters))
    distances = pairwise_distances(X)
    np.fill_diagonal(distances, 0.0 if include_self else np.inf)
    delta = np.partition(distances, count - 1, axis=1)[:, :count].sum(axis=1)
    spread = float(np.ptp(delta))
    if spread <= EPS:
        q = np.full(n, 0.5)
    else:
        q = np.clip((delta - delta.min()) / spread, 0.0, 1.0)
    return 1.0 - q, delta, count


class MultipleCoefficientFuzzyCMeans:
    def __init__(self, n_clusters, max_iter=DEFAULT_MAX_ITER,
                 tol=DEFAULT_TOL, random_state=DEFAULT_RANDOM_STATE, m_lower=1.1,
                 m_upper=4.1, power=1.0, include_self=True, init_centers=None):
        self.n_clusters = int(n_clusters)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.random_state = random_state
        self.m_lower, self.m_upper, self.power = float(m_lower), float(m_upper), float(power)
        self.include_self, self.init_centers = bool(include_self), init_centers
        if self.max_iter < 1:
            raise ValueError('max_iter must be positive.')
        if not (1 < self.m_lower <= self.m_upper and self.power > 0
                and np.isfinite([self.m_lower, self.m_upper, self.power]).all()):
            raise ValueError('MC-FCM requires finite 1<m_lower<=m_upper and power>0.')

    def fuzzifier_values(self, X):
        density, delta, count = neighbor_concentration(X, self.n_clusters, self.include_self)
        power = self.power
        extra = {'mc_power':power, 'mc_neighbors':count, 'mc_include_self': self.include_self,
                 'density_family':'neighbor', 'mapping_family':'power'}
        m = self.m_lower + (self.m_upper - self.m_lower) * (1.0 - density) ** power
        extra.update({'fuzzifier_lower':self.m_lower,'fuzzifier_upper':self.m_upper})
        return m, extra

    def fit(self, X):
        start = perf_counter()
        X = np.asarray(X, dtype=np.float64)
        m, extra = self.fuzzifier_values(X)
        density_time = perf_counter() - start
        rng = np.random.default_rng(self.random_state)
        centers = (X[rng.choice(len(X), self.n_clusters, replace=False)].copy()
                   if self.init_centers is None else np.asarray(self.init_centers, dtype=float).copy())
        if centers.shape != (self.n_clusters, X.shape[1]):
            raise ValueError('Initial center dimensions do not match K/features.')
        converged = False
        iteration_start = perf_counter()
        previous_objective = np.inf
        objective_increases = 0
        for iteration in range(1, self.max_iter + 1):
            old_centers = centers.copy()
            U = update_fcm_memberships_from_d2(distances_to_centers_squared(X, centers), m)
            weights = U ** m[:, None]
            centers = (weights.T @ X) / np.maximum(weights.sum(axis=0)[:, None], EPS)
            delta = float(np.linalg.norm(centers - old_centers, ord='fro'))
            d2 = distances_to_centers_squared(X, centers)
            objective = float(np.sum(U ** m[:, None] * d2))
            if objective > previous_objective + 1e-9 * max(1.,abs(previous_objective)):
                objective_increases += 1
            previous_objective = objective
            if delta < self.tol:
                converged = True
                break
        iteration_time = perf_counter() - iteration_start
        extra.update({'density_time_s':density_time,'iteration_time_s':iteration_time,
                      'm_i_min':float(m.min()),'m_i_mean':float(m.mean()),'m_i_max':float(m.max()),
                      'e_i_min':float((1/(m-1)).min()),'e_i_mean':float((1/(m-1)).mean()),
                      'e_i_max':float((1/(m-1)).max()),'objective_increases':objective_increases,
                      'final_change':delta,
                      'stopping': 'center_Frobenius_norm'})
        return FitResult('MC-FCM', converged, iteration, perf_counter()-start,
                         objective, U, centers, extra)
