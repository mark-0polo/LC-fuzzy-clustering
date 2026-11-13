"""Pure array operations; no file or report dependencies."""
import numpy as np
from ._constants import EPS

def standardize(X):
    X = np.asarray(X, dtype=np.float64)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    return (X - mean) / std


def pairwise_squared_distances(X):
    X = np.asarray(X, dtype=np.float64)
    squared_norms = np.sum(X * X, axis=1)
    d2 = squared_norms[:, None] + squared_norms[None, :] - 2.0 * (X @ X.T)
    return np.maximum(d2, 0.0)


def pairwise_distances(X):
    return np.sqrt(pairwise_squared_distances(X))


def safe_row_normalize(A):
    A = np.asarray(A, dtype=np.float64)
    row_sum = A.sum(axis=1, keepdims=True)
    bad = row_sum[:, 0] <= EPS
    out = A / np.maximum(row_sum, EPS)
    if np.any(bad):
        out[bad] = 1.0 / A.shape[1]
    return out


def update_fcm_memberships_from_d2(d2, m):
    d2 = np.asarray(d2, dtype=np.float64)
    n_samples, n_clusters = d2.shape
    U = np.zeros_like(d2)
    m = np.asarray(m, dtype=np.float64)
    if not np.all(np.isfinite(m)) or np.any(m <= 1):
        raise ValueError("Fuzzifiers must be finite and greater than 1.")
    exponents = np.broadcast_to(1.0 / (m - 1.0), (n_samples,))

    zero_mask = d2 <= EPS
    rows_with_zero = np.any(zero_mask, axis=1)
    if np.any(rows_with_zero):
        counts = np.sum(zero_mask[rows_with_zero], axis=1, keepdims=True)
        U[rows_with_zero] = zero_mask[rows_with_zero] / counts

    nonzero_rows = ~rows_with_zero
    if np.any(nonzero_rows):
        log_weights = -exponents[nonzero_rows, None] * np.log(np.maximum(d2[nonzero_rows], EPS))
        log_weights -= log_weights.max(axis=1, keepdims=True)
        weights = np.exp(log_weights)
        U[nonzero_rows] = weights / weights.sum(axis=1, keepdims=True)

    return U


def distances_to_centers_squared(X, centers):
    return np.column_stack([np.sum((X - center) ** 2, axis=1) for center in centers])


def fcm_centers(X, U, m):
    weights = U ** m
    denominator = np.maximum(weights.sum(axis=0)[:, None], EPS)
    return (weights.T @ X) / denominator


def fcm_objective(X, U, centers, m):
    d2 = distances_to_centers_squared(X, centers)
    return float(np.sum((U ** m) * d2))
