"""Numerical and evaluation defaults."""
import numpy as np

EPS = np.finfo(np.float64).eps
DEFAULT_RANDOM_STATE = 42
DEFAULT_MAX_ITER = 5000
DEFAULT_TOL = 1e-5
DEFAULT_N_SEEDS = 30
STATISTICAL_TEST_ALPHA = 0.05
BENCHMARK_METRICS = ("pc", "pe", "sil", "pur", "nmi", "ari")
METRIC_DIRECTIONS = {
    "pc": True,
    "pe": False,
    "sil": True,
    "pur": True,
    "nmi": True,
    "ari": True,
}
