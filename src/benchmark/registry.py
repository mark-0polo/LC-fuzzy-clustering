"""Model construction and benchmark evaluation, independent of file layout."""
import numpy as np
from algo._constants import (DEFAULT_MAX_ITER, DEFAULT_TOL)
from algo.fcm import FuzzyCMeans
from algo.pcm import PossibilisticCMeans
from algo.pfcm import PossibilisticFuzzyCMeans
from algo.efcm import EntropyRegularizedFCM
from algo.mc_fcm import MultipleCoefficientFuzzyCMeans
from algo.lc_fixed_t import LCFixedT
from algo.lc_fixed_half_t import LCFixedHalfT
from algo.lc_soft_fixed_t import LCSoftFixedT
from metrics import evaluate_memberships
from metrics import (silhouette_score_np, purity_score, normalized_mutual_info,
                                  adjusted_rand_index, partition_coefficient, partition_entropy)
from algo._numerics import pairwise_distances, safe_row_normalize
from algo.dpc import DensityPeaks
from algo.kmeans_pp import KMeansPlusPlus

def make_model(method, n_clusters, seed, max_iter=DEFAULT_MAX_ITER, tol=DEFAULT_TOL, parameters=None):
    from .config import PRIMARY
    if method not in PRIMARY:
        raise ValueError('Unknown or retired method: ' + method)
    parameters = parameters or {}
    options = dict(max_iter=max_iter, tol=tol, random_state=seed)
    if method == 'FCM':
        return FuzzyCMeans(n_clusters, m=2., **options)
    if method == 'PCM':
        return PossibilisticCMeans(n_clusters, m=2., **options)
    if method == 'PFCM':
        return PossibilisticFuzzyCMeans(n_clusters, m=2., nu=2., a=1., b=1., **options)
    if method == 'DPC':
        return DensityPeaks(n_clusters, **parameters)
    if method == 'eFCM':
        return EntropyRegularizedFCM(n_clusters, **options, **parameters)
    if method == 'k-means++':
        return KMeansPlusPlus(n_clusters, random_state=seed)
    if method == 'MC-FCM':
        return MultipleCoefficientFuzzyCMeans(n_clusters, **options, **parameters)
    controls = {'LC-FixedT': LCFixedT, 'LC-FixedHalfT': LCFixedHalfT,
                'LC-SoftFixedT': LCSoftFixedT}
    if parameters:
        raise ValueError('The fixed-temperature controls have no tuning switches.')
    return controls[method](n_clusters, **options)


def run_single_method(dataset, method, seed, max_iter=DEFAULT_MAX_ITER, tol=DEFAULT_TOL, parameters=None):
    model = make_model(method, dataset["n_clusters"], seed, max_iter=max_iter, tol=tol, parameters=parameters)
    result = model.fit(dataset["X"])
    labels = result.labels if result.labels is not None else np.argmax(result.U, axis=1)
    if result.membership_kind == 'fuzzy':
        metrics = evaluate_memberships(dataset['X'], dataset['y'], result.U,
                                       distances=dataset.get('evaluation_distances'))
    else:
        metrics = {'pc': np.nan, 'pe': np.nan,
                   'sil': silhouette_score_np(dataset['X'], labels, dataset.get('evaluation_distances')),
                   'pur': purity_score(dataset['y'], labels),
                   'nmi': normalized_mutual_info(dataset['y'], labels),
                   'ari': adjusted_rand_index(dataset['y'], labels)}
    center_separation = float(pairwise_distances(result.centers).max())
    uniform_deviation = (float(np.max(np.abs(safe_row_normalize(result.U) - 1.0 / dataset['n_clusters'])))
                         if result.U is not None else np.nan)

    row = {
        "dataset": dataset["name"],
        "samples": dataset["samples"],
        "features": dataset["features"],
        "classes": dataset["classes"],
        "requested_clusters": dataset["n_clusters"],
        "method": method,
        "seed": seed,
        "converged": result.converged,
        "iterations": result.iterations if method != 'DPC' else np.nan,
        'membership_kind': result.membership_kind,
        "total_time_s": result.total_time_s,
        "objective": result.objective,
        "hard_clusters": int(np.unique(labels).size),
        "silhouette_defined": bool(np.isfinite(metrics['sil'])),
        "center_max_separation": center_separation,
        "uniform_membership_max_deviation": uniform_deviation,
        "numerically_collapsed": bool(center_separation < 1e-4 and uniform_deviation < 1e-4),
        **metrics,
    }
    row.update(result.extra)
    if result.typicalities is not None:
        normalized = safe_row_normalize(result.typicalities)
        row.update({'normalized_typicality_pc': partition_coefficient(normalized),
                    'normalized_typicality_pe': partition_entropy(normalized)})
    return row


