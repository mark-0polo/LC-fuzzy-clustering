"""Figure text never overlaps or leaves the figure, at the configured fonts and at 30% larger fonts."""
import unittest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import figures
from benchmark.figure_style import FONT
from benchmark.config import PRIMARY, DATASETS, HARD_METHODS
from benchmark.statistics import summarize_repeated_runs
from benchmark.analysis import comparison


def synthetic_results():
    """All 21 datasets and 10 methods, with values in the ranges the real benchmark produces."""
    rng = np.random.default_rng(0)
    rows = []
    for dataset in DATASETS:
        for method in PRIMARY:
            for seed in (42, 43):
                fuzzy = method not in HARD_METHODS + ['PCM']
                rows.append({'dataset': dataset, 'samples': 200, 'features': 5, 'classes': 3, 'method': method,
                             'seed': seed, 'converged': True,
                             'iterations': np.nan if method == 'DPC' else float(rng.integers(5, 1500)),
                             'objective': 1., 'pc': rng.uniform(.1, 1) if fuzzy else np.nan,
                             'pe': rng.uniform(0, 2.7) if fuzzy else np.nan, 'sil': rng.uniform(-.4, .6),
                             'pur': rng.uniform(.1, 1), 'nmi': rng.uniform(0, .9), 'ari': rng.uniform(-.12, .93),
                             'total_time_s': rng.uniform(.001, 3), 'native_reported_time_s': .01,
                             'm_i_min': 1.1, 'm_i_mean': 2., 'm_i_max': 4.1})
    frame = pd.DataFrame(rows)
    return summarize_repeated_runs(frame), comparison(frame, PRIMARY)[0], frame


class FigureLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = synthetic_results()

    def problems(self):
        found = {}
        with plt.rc_context(figures.style()):
            for name, fig in figures.build_figures(self.results[0], PRIMARY, self.results[1], self.results[2]):
                issues = figures.text_layout_problems(fig)
                plt.close(fig)
                if issues:
                    found[name] = issues[:3]
        return found

    def test_checker_detects_overlap_and_clipping(self):
        fig = figures.new_figure(4, 3)
        ax = fig.subplots()
        ax.text(.5, .5, 'AAAA', transform=ax.transAxes)
        ax.text(.52, .5, 'BBBB', transform=ax.transAxes)
        fig.text(.99, .5, 'falls off the edge')
        try:
            issues = figures.text_layout_problems(fig)
        finally:
            plt.close(fig)
        self.assertTrue(any(i.startswith('overlap') for i in issues))
        self.assertTrue(any(i.startswith('outside') for i in issues))

    def test_no_text_problems_at_configured_fonts(self):
        self.assertEqual(self.problems(), {})

    def test_no_text_problems_with_fonts_30_percent_larger(self):
        original = dict(FONT)
        try:
            FONT.update({key: value * 1.3 for key, value in original.items()})
            self.assertEqual(self.problems(), {})
        finally:
            FONT.update(original)


if __name__ == '__main__':
    unittest.main()
