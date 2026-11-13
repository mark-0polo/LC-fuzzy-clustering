"""RESULTS.md tables, winners and the LC-vs-others significance tests on synthetic fits."""
import tempfile
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from benchmark.config import PRIMARY, DATASETS, LC_METHODS
from benchmark.results_markdown import write_results, cell_means, winners
from benchmark.statistics import proposed_versus_others_tests


def synthetic_runs():
    rows = []
    for d_index, dataset in enumerate(DATASETS):
        for seed in (42, 43):
            for m_index, method in enumerate(PRIMARY):
                # LC-2T is always best; the small dataset term avoids all-tied differences.
                value = .1 + .05 * m_index * (1 + .01 * d_index) + (.3 if method == 'LC-SoftFixedT' else 0.)
                rows.append({'dataset': dataset, 'method': method, 'seed': seed, 'samples': 10 + d_index,
                             'features': 3, 'classes': 2, 'hard_clusters': 1 if method == 'PCM' else 2,
                             'converged': True, 'total_time_s': .01, 'ari': value, 'nmi': value, 'pur': value,
                             'sil': np.nan if method == 'PCM' else value, 'pc': value, 'pe': 1 - value})
    return pd.DataFrame(rows)


class ResultsMarkdownTests(unittest.TestCase):
    def test_tables_winners_and_sections(self):
        frame = synthetic_runs()
        table = cell_means(frame, 'ari')
        self.assertEqual(list(table.index), DATASETS)
        self.assertEqual(winners(table.iloc[0]), ['LC-SoftFixedT'])
        self.assertTrue(cell_means(frame, 'sil')['PCM'].isna().all())      # undefined stays undefined
        with tempfile.TemporaryDirectory() as directory:
            significance = write_results(frame, directory, figures=24)
            text = (Path(directory) / 'RESULTS.md').read_text(encoding='utf-8')
        for heading in ['## 1. Overall ranking', '## 4. Dataset × method tables', '## 6. Is the difference significant?',
                        '## 8. Run health', '## 10. Appendix', '## 12. What can be claimed']:
            self.assertIn(heading, text)
        self.assertIn(f'| 1 | LC-2T | ', text)
        self.assertIn('| PCM | 42 | 0 | 42 | 21 |', text)                  # PCM's single cluster is reported
        self.assertEqual(set(significance.proposed), set(LC_METHODS))

    def test_proposed_versus_others_direction_and_holm_family(self):
        frame = synthetic_runs()
        others = [m for m in PRIMARY if m not in LC_METHODS]
        tests = proposed_versus_others_tests(frame, LC_METHODS, others, DATASETS, metrics=('ari',))
        self.assertEqual(len(tests), len(LC_METHODS) * len(others))
        best = tests[tests.proposed == 'LC-SoftFixedT']
        self.assertTrue((best.proposed_better == len(DATASETS)).all())
        self.assertTrue(best.significant_holm_0_05.all())
        self.assertTrue((tests.p_holm >= tests.p_value).all())


if __name__ == '__main__':
    unittest.main()
