"""The 21 bundled UCI datasets: file parsing, label coding and feature standardization.

Every dataset is read from data/uci/ (no downloads); data/checksums.json holds the SHA-256 of
every raw file and is checked before each run. Labels and ID columns never enter X. Each
dataset is standardized feature-wise (population SD) and K is the number of true classes.

Groups (kept in every report so readers can see what was used when):
  development  the 8 datasets used while the methods were developed
  regime       5 datasets added to test where the LC methods should win or lose
  held-out     8 datasets that were not looked at before their first full run
"""
from pathlib import Path
import numpy as np
import pandas as pd
from algo._numerics import standardize


def _csv(path, **options):
    return pd.read_csv(path, **options).dropna()


def _iris(d):
    cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
    df = _csv(d / 'iris.data', names=cols)
    return df[cols[:-1]].to_numpy(), df['class']


def _wine(d):
    cols = [f'feature_{i}' for i in range(1, 14)]
    df = _csv(d / 'wine.data', names=['class'] + cols)
    return df[cols].to_numpy(), df['class']


def _breast_cancer(d):
    cols = [f'feature_{i}' for i in range(1, 31)]
    df = _csv(d / 'wdbc.data', names=['id', 'diagnosis'] + cols)
    return df[cols].to_numpy(), df['diagnosis']


def _seeds(d):
    cols = [f'feature_{i}' for i in range(1, 8)]
    df = _csv(d / 'seeds_dataset.txt', sep=r'\s+', names=cols + ['class'], engine='python')
    return df[cols].to_numpy(), df['class']


def _glass(d):
    cols = ['ri', 'na', 'mg', 'al', 'si', 'k', 'ca', 'ba', 'fe']
    df = _csv(d / 'glass.data', names=['id'] + cols + ['class'])
    return df[cols].to_numpy(), df['class']


def _protein(filename, n_features):
    def read(d):
        cols = [f'feature_{i}' for i in range(1, n_features + 1)]
        df = _csv(d / filename, sep=r'\s+', names=['sequence_name'] + cols + ['class'], engine='python')
        return df[cols].to_numpy(), df['class']
    return read


SEGMENT_FEATURES = ['region_centroid_col', 'region_centroid_row', 'region_pixel_count',
                    'short_line_density_5', 'short_line_density_2', 'vedge_mean', 'vedge_sd',
                    'hedge_mean', 'hedge_sd', 'intensity_mean', 'rawred_mean', 'rawblue_mean',
                    'rawgreen_mean', 'exred_mean', 'exblue_mean', 'exgreen_mean', 'value_mean',
                    'saturation_mean', 'hue_mean']


def _image_segmentation(d):
    names = ['class'] + SEGMENT_FEATURES
    df = pd.concat([pd.read_csv(d / f, skiprows=5, names=names) for f in ('segmentation.data', 'segmentation.test')],
                   ignore_index=True).dropna()
    return df[SEGMENT_FEATURES].to_numpy(), df['class']


def _last_column(filename, delimiter=None):
    """Numeric file whose last column is the class."""
    def read(d):
        a = np.loadtxt(d / filename, delimiter=delimiter)
        return a[:, :-1], a[:, -1]
    return read


def _mfeat_kar(d):
    return np.loadtxt(d / 'mfeat-kar'), np.repeat(np.arange(10), 200)   # 200 rows per digit, in order


def _new_thyroid(d):
    a = np.loadtxt(d / 'new-thyroid.data', delimiter=',')
    return a[:, 1:], a[:, 0]


def _pandas_last_column(filename, **options):
    def read(d):
        df = pd.read_csv(d / filename, header=None, **options)
        return df.iloc[:, :-1].to_numpy(float), df.iloc[:, -1].to_numpy()
    return read


def _vehicle(d):
    df = pd.concat([pd.read_csv(d / f'xa{c}.dat', sep=r'\s+', header=None) for c in 'abcdefghi'], ignore_index=True)
    return df.iloc[:, :18].to_numpy(float), df.iloc[:, 18].to_numpy()


def _parkinsons(d):
    df = pd.read_csv(d / 'parkinsons.data')
    return df.drop(columns=['name', 'status']).to_numpy(float), df['status'].to_numpy()


# (name, group, reader). The order here is the order of every table and figure.
CATALOGUE = [
    ('Iris', 'development', _iris),
    ('Wine', 'development', _wine),
    ('Breast Cancer', 'development', _breast_cancer),
    ('Seeds', 'development', _seeds),
    ('Glass', 'development', _glass),
    ('Ecoli', 'development', _protein('ecoli.data', 7)),
    ('Yeast', 'development', _protein('yeast.data', 8)),
    ('Image Segmentation', 'development', _image_segmentation),
    ('optdigits', 'regime', _last_column('optdigits.tra', ',')),
    ('pendigits', 'regime', _last_column('pendigits.tes', ',')),
    ('satimage', 'regime', _last_column('sat.trn')),
    ('mfeat-kar', 'regime', _mfeat_kar),
    ('new-thyroid', 'regime', _new_thyroid),
    ('banknote', 'held-out', _last_column('data_banknote_authentication.txt', ',')),
    ('wall-following', 'held-out', _pandas_last_column('sensor_readings_4.data')),
    ('wifi', 'held-out', _last_column('wifi_localization.txt')),
    ('vehicle', 'held-out', _vehicle),
    ('parkinsons', 'held-out', _parkinsons),
    ('ionosphere', 'held-out', _pandas_last_column('ionosphere.data')),
    ('sonar', 'held-out', _pandas_last_column('sonar.all-data')),
    ('libras', 'held-out', _last_column('movement_libras.data', ',')),
]
DATASET_NAMES = [name for name, _, _ in CATALOGUE]
DATASET_GROUPS = {name: group for name, group, _ in CATALOGUE}


def label_codes(values):
    """Classes coded 0..K-1 in sorted order of the original labels."""
    return np.unique(np.asarray(values), return_inverse=True)[1].astype(np.int64)


def load_uci_datasets(data_dir, names=None):
    """Return the datasets (all 21 by default) as dicts with standardized X, coded y and K."""
    data_dir = Path(data_dir)
    datasets = []
    for name, group, reader in CATALOGUE:
        if names is not None and name not in names:
            continue
        X, y = reader(data_dir)
        X, y = standardize(np.asarray(X, dtype=float)), label_codes(y)
        classes = int(y.max() + 1)
        datasets.append({'name': name, 'group': group, 'X': X, 'y': y, 'classes': classes,
                         'n_clusters': classes, 'samples': int(X.shape[0]), 'features': int(X.shape[1])})
    return datasets
