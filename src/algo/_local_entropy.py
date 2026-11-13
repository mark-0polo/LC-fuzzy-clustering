"""Three fixed-temperature entropy controls on one frozen local-mean geometry.

These are calibrated controls based on established entropy clustering, rather
than an adaptive log-confidence algorithm. See docs/ALGORITHMS.md.
"""
from time import perf_counter
import numpy as np
from scipy.spatial import cKDTree
from ._models import FitResult

MARGINS = {'LC-FixedT': 2., 'LC-FixedHalfT': 4., 'LC-SoftFixedT': 1.}
EPS = np.finfo(float).eps


def local_coordinates(X):
    unique, inverse, counts = np.unique(X, axis=0, return_inverse=True, return_counts=True)
    if len(unique) < 2:
        return X-X.mean(axis=0)
    k = min(len(unique)-1, max(2, int(np.ceil(np.log2(len(unique))))))
    tree = cKDTree(unique)
    query_count = min(len(unique), k+2)
    distances, indices = tree.query(unique, k=query_count, workers=1)
    selected = indices[:, 1:k+1]
    d2 = np.sum((unique[selected]-unique[:, None, :])**2, axis=2)
    selected = np.take_along_axis(selected, np.lexsort((selected, d2), axis=1), axis=1)
    weights = counts[selected]
    averages = (counts[:, None]*unique+np.einsum('nk,nkd->nd', weights, unique[selected]))/(counts+weights.sum(1))[:, None]
    smoothed = .5*(unique+averages)
    ties = (np.flatnonzero(np.isclose(distances[:, k], distances[:, k+1], rtol=1e-13, atol=0.))
            if query_count > k+1 else [])
    for a in ties:
        boundary = np.nextafter(distances[a, -1], np.inf)
        candidates = np.unique(np.r_[np.asarray(tree.query_ball_point(unique[a], boundary), dtype=int), indices[a]])
        candidates = candidates[candidates != a]
        squared = np.sum((unique[candidates]-unique[a])**2, axis=1)
        selected = candidates[np.lexsort((candidates, squared))[:k]]
        weight = counts[selected]
        average = (counts[a]*unique[a]+weight@unique[selected])/(counts[a]+weight.sum())
        smoothed[a] = .5*(unique[a]+average)
    Q = smoothed[inverse]
    return Q-Q.mean(axis=0)


def calibrated_temperature(Q, margin):
    centered = Q-Q.mean(axis=0)
    singular = np.linalg.svd(centered, compute_uv=False)
    cutoff = max(centered.shape)*EPS*max(float(singular[0]), EPS)
    rank = int(np.count_nonzero(singular > cutoff))
    radius = np.sum(centered*centered, axis=1)
    if rank == 0 or radius.mean() == 0:
        return np.zeros(len(Q)), {'rank': 0, 'uniform_trace_bound': np.nan,
                                  'uniform_spectral_radius': np.nan}
    guarded = np.full(len(radius), radius.mean())
    h = float(np.mean(radius/guarded))
    temperature = 2.*h*guarded/(margin*(rank+2))
    return temperature, {'rank': rank, 'uniform_trace_bound': float(2.*radius.mean()/temperature[0]/rank),
                         'uniform_spectral_radius': float(2.*singular[0]**2/(len(Q)*temperature[0]))}


def squared_distances(Q, V, point_norms):
    return np.maximum(point_norms[:, None]+np.einsum('ij,ij->i', V, V)[None, :]-2*(Q@V.T), 0.)


class LocalEntropyCMeans:
    """Fit one of the three retained controls to standardized feature rows."""
    def __init__(self, n_clusters, method='LC-FixedHalfT', random_state=42, max_iter=5000, tol=1e-5):
        if method not in MARGINS:
            raise ValueError('Choose LC-FixedT, LC-FixedHalfT or LC-SoftFixedT.')
        self.n_clusters, self.method = int(n_clusters), method
        self.random_state, self.max_iter, self.tol = random_state, int(max_iter), float(tol)

    def fit(self, X):
        started = perf_counter()
        X = np.asarray(X, dtype=float)
        if (X.ndim != 2 or min(X.shape) < 1 or not np.isfinite(X).all()
                or not 1 <= self.n_clusters <= len(X) or self.max_iter < 1
                or not np.isfinite(self.tol) or self.tol <= 0):
            raise ValueError('Finite nonempty observations, valid K/budget and finite positive tolerance required.')
        Q = local_coordinates(X)
        self.coordinates_ = Q
        point_norms = np.einsum('ij,ij->i', Q, Q)
        T, details = calibrated_temperature(Q, MARGINS[self.method])
        self.temperatures_ = T
        preparation = perf_counter()-started
        if details['rank'] == 0:
            U = np.full((len(X), self.n_clusters), 1/self.n_clusters)
            V = np.repeat(Q.mean(axis=0)[None, :], self.n_clusters, axis=0)
            self.objective_history_ = np.array([0.])
            return FitResult(self.method, True, 0, perf_counter()-started, 0., U, V,
                             {**details, 'degenerate': True, 'preprocessing_time_s': preparation})
        U = np.random.default_rng(self.random_state).random((len(Q), self.n_clusters))
        U /= U.sum(axis=1, keepdims=True)
        V, history, converged, empty_moves = None, [], False, 0
        for iteration in range(1, self.max_iter+1):
            previous_u, previous_v = U, V
            mass = U.sum(axis=0)
            V = (U.T@Q)/np.maximum(mass, np.finfo(float).tiny)[:, None]
            empty = np.flatnonzero(mass == 0)
            if len(empty):
                losses = squared_distances(Q, V[mass > 0], point_norms).min(axis=1)
                for k in empty:
                    point = int(np.argmax(losses))
                    V[k] = Q[point]
                    losses = np.minimum(losses, np.sum((Q-Q[point])**2, axis=1))
                    empty_moves += 1
            d2 = squared_distances(Q, V, point_norms)
            logits = -(d2-d2.min(axis=1, keepdims=True))/T[:, None]
            weights = np.exp(logits)
            denominator = weights.sum(axis=1)
            U = weights/denominator[:, None]
            entropy = np.log(denominator)-np.einsum('ij,ij->i', U, logits)
            objective = float(np.sum(U*d2)+T@(-entropy))
            if history and objective-history[-1] > 1e-10*max(1., abs(history[-1])):
                raise FloatingPointError('Fixed entropy objective increased.')
            history.append(objective)
            du = float(np.linalg.norm(U-previous_u))
            dv = np.inf if previous_v is None else float(np.linalg.norm(V-previous_v)/max(1., np.linalg.norm(previous_v)))
            if du < self.tol and dv < self.tol:
                converged = True
                break
        self.objective_history_ = np.asarray(history)
        details.update({'temperature': float(T[0]), 'temperature_margin': MARGINS[self.method],
                        'preprocessing_time_s': preparation, 'optimization_time_s': perf_counter()-started-preparation,
                        'final_membership_change': du, 'final_center_change': dv, 'empty_center_moves': empty_moves})
        return FitResult(self.method, converged, iteration, perf_counter()-started, objective, U, V, details)
