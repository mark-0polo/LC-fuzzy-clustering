"""Clustering validity metrics over arrays."""
import numpy as np
from algo._constants import EPS
from algo._numerics import pairwise_distances, safe_row_normalize

def partition_coefficient(U):
    return float(np.sum(U ** 2) / U.shape[0])


def partition_entropy(U):
    return float(-np.sum(U * np.log(np.maximum(U, EPS))) / U.shape[0])


def contingency_matrix(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    true_values, true_inv = np.unique(y_true, return_inverse=True)
    pred_values, pred_inv = np.unique(y_pred, return_inverse=True)
    table = np.zeros((len(true_values), len(pred_values)), dtype=np.int64)
    np.add.at(table, (true_inv, pred_inv), 1)
    return table


def purity_score(y_true, y_pred):
    table = contingency_matrix(y_true, y_pred)
    return float(table.max(axis=0).sum() / table.sum())


def normalized_mutual_info(y_true, y_pred):
    table = contingency_matrix(y_true, y_pred).astype(np.float64)
    n = table.sum()
    if n == 0:
        return 0.0

    p = table / n
    p_true = p.sum(axis=1, keepdims=True)
    p_pred = p.sum(axis=0, keepdims=True)

    expected = p_true @ p_pred
    mask = p > 0
    mi = float(np.sum(p[mask] * np.log(p[mask] / expected[mask])))

    h_true = float(-np.sum(p_true[p_true > 0] * np.log(p_true[p_true > 0])))
    h_pred = float(-np.sum(p_pred[p_pred > 0] * np.log(p_pred[p_pred > 0])))
    denom = h_true + h_pred
    if denom <= EPS:
        return 1.0
    return float(2.0 * mi / denom)


def comb2(x):
    x = np.asarray(x, dtype=np.float64)
    return x * (x - 1.0) / 2.0


def adjusted_rand_index(y_true, y_pred):
    table = contingency_matrix(y_true, y_pred).astype(np.float64)
    n = table.sum()
    if n < 2:
        return 1.0

    sum_comb = comb2(table).sum()
    row_comb = comb2(table.sum(axis=1)).sum()
    col_comb = comb2(table.sum(axis=0)).sum()
    total_comb = comb2(n)
    expected = row_comb * col_comb / total_comb
    max_index = 0.5 * (row_comb + col_comb)
    denom = max_index - expected
    if abs(denom) <= EPS:
        return 1.0
    return float((sum_comb - expected) / denom)


def silhouette_score_np(X, labels, distances=None):
    X = np.asarray(X, dtype=np.float64)
    labels = np.asarray(labels)
    unique, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
    if unique.size <= 1 or unique.size >= X.shape[0]:
        return np.nan
    distances = pairwise_distances(X) if distances is None else np.asarray(distances)
    means = np.column_stack([distances[:, inverse == j].sum(axis=1) / counts[j]
                             for j in range(unique.size)])
    row = np.arange(X.shape[0])
    own_count = counts[inverse]
    a = (means[row, inverse] * own_count - np.diag(distances)) / np.maximum(own_count - 1, 1)
    means[row, inverse] = np.inf
    b = means.min(axis=1)
    scores = (b - a) / np.maximum(np.maximum(a, b), EPS)
    scores[own_count == 1] = 0.0
    return float(scores.mean())


def evaluate_memberships(X, y_true, U, distances=None):
    U = safe_row_normalize(U)
    labels = np.argmax(U, axis=1)
    return {
        "pc": partition_coefficient(U),
        "pe": partition_entropy(U),
        "sil": silhouette_score_np(X, labels, distances=distances),
        "pur": purity_score(y_true, labels),
        "nmi": normalized_mutual_info(y_true, labels),
        "ari": adjusted_rand_index(y_true, labels),
    }
