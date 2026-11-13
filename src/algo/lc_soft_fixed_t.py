"""LC-SoftFixedT: local-mean geometry and fixed entropy temperature 2T.

The common objective and block updates live in _local_entropy.py; this file
selects this method's fixed temperature without duplicating its equations.
"""
from ._local_entropy import LocalEntropyCMeans


class LCSoftFixedT(LocalEntropyCMeans):
    """Local-mean clustering with temperature 2T, frozen during fitting."""
    def __init__(self, n_clusters, random_state=42, max_iter=5000, tol=1e-5):
        super().__init__(n_clusters, method='LC-SoftFixedT', random_state=random_state,
                         max_iter=max_iter, tol=tol)
