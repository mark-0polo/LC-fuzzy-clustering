"""k-means++: Lloyd's k-means with k-means++ seeding (scikit-learn), one start per seed.

Standard non-fuzzy baseline. Library defaults: max_iter=300, tol=1e-4, n_init=1, so every
method gets one random start per benchmark seed. Returns hard labels; PC/PE are undefined.
References: Lloyd (1982), https://doi.org/10.1109/TIT.1982.1056489; Arthur & Vassilvitskii
(2007), k-means++.
"""
from time import perf_counter
import numpy as np
from sklearn.cluster import KMeans
from ._models import FitResult


class KMeansPlusPlus:
    def __init__(self, n_clusters, random_state=42):
        self.n_clusters, self.random_state = int(n_clusters), random_state

    def fit(self, X):
        started = perf_counter()
        X = np.asarray(X, dtype=float)
        model = KMeans(self.n_clusters, n_init=1, random_state=self.random_state).fit(X)
        return FitResult('k-means++', bool(model.n_iter_ < model.max_iter), int(model.n_iter_),
                         perf_counter() - started, float(model.inertia_), None, model.cluster_centers_,
                         {'library': 'scikit-learn KMeans', 'n_init': 1},
                         membership_kind='hard', labels=model.labels_.astype(int))
