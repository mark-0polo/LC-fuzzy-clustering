"""The fixed benchmark protocol: 10 methods, 21 datasets, 30 seeds."""
from data_loader import DATASET_NAMES, DATASET_GROUPS

# The three LC variants are the proposed methods; the other seven are comparison methods.
LC_METHODS = ['LC-SoftFixedT', 'LC-FixedT', 'LC-FixedHalfT']
PRIMARY = ['FCM', 'PCM', 'PFCM', 'DPC', 'eFCM', 'MC-FCM',
           'LC-FixedT', 'LC-FixedHalfT', 'LC-SoftFixedT', 'k-means++']
DATASETS = list(DATASET_NAMES)
GROUPS = dict(DATASET_GROUPS)
# The 4 regime datasets chosen because the LC methods were predicted to win there
# (new-thyroid, the fifth regime dataset, was a predicted-loss control).
SELECTED_FOR_LC = ['optdigits', 'pendigits', 'satimage', 'mfeat-kar']

PROTOCOL_VERSION = 'ten_methods_twentyone_datasets_v3'
DEFAULT_OUTPUT = 'results/benchmark'
SEEDS_START = 42
HARD_METHODS = ['DPC', 'k-means++']
# Wins and the ranking score use only the label-agreement metrics. PC/PE measure membership
# sharpness, which a lower temperature increases by construction; Silhouette uses no labels.
LABEL_METRICS = ['ari', 'nmi', 'pur']


def metric_methods(metric, methods=PRIMARY):
    """Methods for which a metric is defined (PC/PE need native fuzzy memberships)."""
    return [m for m in methods if metric not in ('pc', 'pe') or m not in HARD_METHODS + ['PCM']]
