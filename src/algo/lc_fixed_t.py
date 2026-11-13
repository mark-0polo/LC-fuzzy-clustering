"""LC-FixedT: local-mean geometry and fixed entropy temperature T.

The common objective and block updates live in _local_entropy.py; this file
selects this method's fixed temperature without duplicating its equations.
"""
from ._local_entropy import LocalEntropyCMeans


class LCFixedT(LocalEntropyCMeans):
    """Local-mean clustering with temperature T, frozen during fitting."""
    def __init__(self, n_clusters, random_state=42, max_iter=5000, tol=1e-5):
        super().__init__(n_clusters, method='LC-FixedT', random_state=random_state,
                         max_iter=max_iter, tol=tol)
