"""Author-source DPC computations with an explicitly automated known-K selector.

Laio's cluster_dp.m supplies Gaussian density, order-statistic cutoff, ranked
nearest-denser assignment and halo. Choosing top-K rho*delta replaces the
interactive decision-graph rectangle; it is a disclosed benchmark adaptation.
"""
from time import perf_counter
import numpy as np
from ._constants import EPS
from ._models import FitResult
from ._numerics import pairwise_distances


def density_peak_quantities(X, percent=2.0):
    X = np.asarray(X, dtype=float)
    if len(X) < 2 or not 0 < percent <= 100:
        raise ValueError('DPC requires at least two samples and percent in (0,100].')
    distances = pairwise_distances(X)
    np.fill_diagonal(distances, 0.0)
    upper = distances[np.triu_indices(len(X), 1)]
    # MATLAB round for positive arguments, followed by one-based indexing.
    position = min(len(upper), max(1, int(np.floor(len(upper) * percent / 100 + 0.5))))
    dc = float(np.partition(upper, position - 1)[position - 1])
    if dc <= EPS:
        positive = upper[upper > EPS]
        dc = float(positive.min()) if len(positive) else EPS
    kernel = np.exp(-(distances / dc) ** 2)
    np.fill_diagonal(kernel, 0.0)
    rho = kernel.sum(axis=1)
    order = np.argsort(-rho, kind='stable')
    delta = np.zeros(len(X))
    nearest = np.full(len(X), -1, dtype=int)
    for rank, sample in enumerate(order[1:], 1):
        higher = order[:rank]
        neighbor = higher[np.argmin(distances[sample, higher])]
        nearest[sample] = neighbor
        delta[sample] = distances[sample, neighbor]
    delta[order[0]] = delta[order[1:]].max()
    return distances, dc, rho, delta, order, nearest


def select_known_k_centers(rho, delta, n_clusters):
    if not 1 <= n_clusters <= len(rho):
        raise ValueError('DPC requires 1<=K<=N.')
    centers = np.argsort(-(rho * delta), kind='stable')[:n_clusters]
    peak = int(np.argmax(rho))
    # Guard zero-score degeneracy; every assignment chain must reach a center.
    if peak not in centers:
        centers[-1] = peak
    return centers


def assign_density_peaks(distances, dc, rho, order, nearest, centers):
    labels = np.full(len(rho), -1, dtype=int)
    labels[centers] = np.arange(len(centers))
    for sample in order:
        if labels[sample] < 0:
            labels[sample] = labels[nearest[sample]]
    first, second = np.triu_indices(len(rho), 1)
    boundary = (labels[first] != labels[second]) & (distances[first, second] <= dc)
    first, second = first[boundary], second[boundary]
    border_density = np.zeros(len(centers))
    average = (rho[first] + rho[second]) / 2
    np.maximum.at(border_density, labels[first], average)
    np.maximum.at(border_density, labels[second], average)
    halo = labels.copy()
    halo[rho < border_density[labels]] = -1
    return labels, halo


class DensityPeaks:
    def __init__(self, n_clusters, percent=2.0):
        self.n_clusters, self.percent = int(n_clusters), float(percent)

    def fit(self, X):
        start = perf_counter()
        X = np.asarray(X, dtype=float)
        distances, dc, rho, delta, order, nearest = density_peak_quantities(X, self.percent)
        centers = select_known_k_centers(rho, delta, self.n_clusters)
        labels, halo = assign_density_peaks(distances, dc, rho, order, nearest, centers)
        return FitResult('DPC', True, 1, perf_counter() - start, np.nan, None, X[centers],
                         {'dpc_dc': dc, 'dpc_percent': self.percent, 'halo_samples': int((halo < 0).sum()),
                          'assignment_passes': 1, 'center_selection': 'known_K_top_rho_delta',
                          'native_output': 'hard_labels'}, membership_kind='hard', labels=labels)
