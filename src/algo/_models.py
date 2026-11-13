"""Result type returned by clustering algorithms."""
from dataclasses import dataclass
import numpy as np

@dataclass
class FitResult:
    method: str
    converged: bool
    iterations: int
    total_time_s: float
    objective: float
    U: np.ndarray | None
    centers: np.ndarray
    extra: dict
    membership_kind: str = 'fuzzy'
    labels: np.ndarray | None = None
    typicalities: np.ndarray | None = None
