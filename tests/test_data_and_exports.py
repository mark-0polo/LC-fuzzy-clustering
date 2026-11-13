"""Bundled data, heatmap labels and high-resolution PNG export checks."""
import hashlib
import io
import json
from pathlib import Path
import unittest

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from figures import STYLE, draw_heatmap, metric_matrix, save_png
from benchmark.config import PRIMARY, DATASETS
from data_loader import load_uci_datasets

ROOT = Path(__file__).resolve().parents[1]


class DataAndExportTests(unittest.TestCase):
    def test_bundled_files_preserve_original_bytes(self):
        expected = json.loads((ROOT / 'data/checksums.json').read_text(encoding='utf-8'))
        self.assertEqual(len(expected), 30)
        for filename, digest in expected.items():
            self.assertEqual(hashlib.sha256((ROOT / 'data/uci' / filename).read_bytes()).hexdigest(), digest)

    def test_loader_dimensions_and_standardization(self):
        dimensions = [(150, 4, 3), (178, 13, 3), (569, 30, 2), (210, 7, 3),
                      (214, 9, 6), (336, 7, 8), (1484, 8, 10), (2310, 19, 7),
                      (3823, 64, 10), (3498, 16, 10), (4435, 36, 6), (2000, 64, 10), (215, 5, 3),
                      (1372, 4, 2), (5456, 4, 4), (2000, 7, 4), (846, 18, 4), (195, 22, 2),
                      (351, 34, 2), (208, 60, 2), (360, 90, 15)]
        datasets = load_uci_datasets(ROOT / 'data/uci')
        self.assertEqual([item['name'] for item in datasets], DATASETS)
        for item, (rows, features, classes) in zip(datasets, dimensions):
            self.assertEqual(item['X'].shape, (rows, features))
            self.assertEqual(len(np.unique(item['y'])), classes)
            self.assertTrue(np.isfinite(item['X']).all())
            np.testing.assert_allclose(item['X'].mean(axis=0), 0, atol=1e-10)
            deviations = item['X'].std(axis=0)
            np.testing.assert_allclose(deviations[deviations > 1e-10], 1, atol=1e-10)
        segmentation = datasets[DATASETS.index('Image Segmentation')]['X']
        self.assertEqual(len(segmentation) - len(np.unique(segmentation, axis=0)), 224)

    def test_heatmap_orientation_is_methods_by_datasets(self):
        records = [{'method': method, 'dataset': dataset, 'pc_mean': 10 * row + col}
                   for row, method in enumerate(PRIMARY) for col, dataset in enumerate(DATASETS)]
        frame = pd.DataFrame(records).sample(frac=1, random_state=9)
        matrix = metric_matrix(frame, 'pc')
        self.assertEqual(matrix.shape, (len(PRIMARY), len(DATASETS)))
        self.assertEqual(matrix[PRIMARY.index('MC-FCM'), DATASETS.index('Wine')], 51)
        self.assertEqual(matrix[PRIMARY.index('LC-FixedHalfT'), DATASETS.index('Yeast')], 10*PRIMARY.index('LC-FixedHalfT')+6)

    def test_png_has_600_dpi_and_records_actual_displayed_values_and_positions(self):
        n = len(DATASETS)
        matrix = np.linspace(-0.2, 0.8, len(PRIMARY)*n).reshape(len(PRIMARY), n)
        matrix[2, 1] = np.nan
        with plt.rc_context(STYLE):
            fig, ax = plt.subplots(figsize=(5, 3))
            try:
                mesh = draw_heatmap(fig, ax, matrix, 'sil', DATASETS, compact=True)
                self.assertTrue(np.ma.getmaskarray(mesh.get_array())[2, 1])
                output = io.BytesIO()
                save_png(fig, output, heatmap=True)
                output.seek(0)
                with Image.open(output) as png:
                    self.assertEqual(png.format, 'PNG')
                    self.assertGreater(min(png.info['dpi']), 599)
                    labels = json.loads(png.info['HeatmapAnnotations'])
                    self.assertEqual(len(labels), len(PRIMARY)*n)
                    for r in range(len(PRIMARY)):
                        for c in range(n):
                            value = matrix[r,c]
                            self.assertEqual(labels[r*n+c], {'text': f'{value:.3f}' if np.isfinite(value) else '--',
                                                            'position': [c+.5,r+.5]})
                    png.verify()
            finally:
                plt.close(fig)


if __name__ == '__main__':
    unittest.main()
