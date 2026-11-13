"""Independent numerical and artist checks for the complete reference plot suite."""
import unittest
from unittest.mock import Mock
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from benchmark.config import PRIMARY, DATASETS
from benchmark.figure_data import (FIGURE_NAMES, relative_percent, amortized_iteration_costs,
                                   iteration_reductions, relative_metric_changes)
from figures import (metric_bars, grouped_bars, capture_plot_data, pipeline_figure, save_png,
                     standalone_bar_figure, DATASET_ROWS, heatmap_winners)


class ReferenceFigureTests(unittest.TestCase):
    def test_failed_png_render_preserves_existing_file_and_removes_temporary_output(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'plot.png'
            path.write_bytes(b'previous image')
            figure=Mock()
            figure.savefig.side_effect=OSError('interrupted PNG write')
            with self.assertRaises(OSError):save_png(figure,path)
            self.assertEqual(path.read_bytes(),b'previous image')
            self.assertEqual(list(Path(directory).iterdir()),[path])

    def test_atomic_png_export_replaces_existing_file_with_valid_image(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'plot.png';path.write_bytes(b'previous image')
            fig,ax=plt.subplots(figsize=(2,2));ax.plot([0,1],[0,1])
            try:save_png(fig,path)
            finally:plt.close(fig)
            with Image.open(path) as png:
                self.assertEqual(png.format,'PNG')
                self.assertGreater(min(png.info['dpi']),599)
                png.verify()
            self.assertEqual(list(Path(directory).iterdir()),[path])

    def test_relative_percent_handles_direction_negative_reference_and_zero(self):
        np.testing.assert_allclose(relative_percent([1.5,-1.,1.],[2.,-2.,0.]),[-25.,50.,np.nan],equal_nan=True)
        np.testing.assert_allclose(relative_percent([1.5],[2.],True),[25.])

    def test_iteration_cost_averages_fit_ratios_and_excludes_non_iterative_or_failed_fits(self):
        rows=[{'dataset':'Iris','method':'FCM','converged':True,'iterations':2,'total_time_s':1.},
              {'dataset':'Iris','method':'FCM','converged':True,'iterations':3,'total_time_s':9.},
              {'dataset':'Iris','method':'FCM','converged':False,'iterations':1,'total_time_s':100.},
              {'dataset':'Iris','method':'FCM','converged':True,'iterations':0,'total_time_s':100.},
              {'dataset':'Iris','method':'DPC','converged':True,'iterations':1,'total_time_s':100.}]
        costs=amortized_iteration_costs(pd.DataFrame(rows),['FCM','DPC'])
        self.assertEqual(list(costs.columns),['FCM'])
        self.assertEqual(costs.loc['Iris','FCM'],1750.)
        self.assertTrue(np.isnan(costs.loc['Wine','FCM']))

    def test_relative_changes_use_equal_weight_dataset_percentages(self):
        rows=[]
        for dataset in DATASETS:
            for method,value in [('FCM',2.),('LC-FixedHalfT',3.)]:
                row={'dataset':dataset,'method':method,'iterations_mean':value*10}
                row.update({m+'_mean':value for m in ['pc','pe','sil','pur','nmi','ari']})
                rows.append(row)
        summary=pd.DataFrame(rows)
        changes=relative_metric_changes(summary)
        self.assertEqual(changes.loc['pc','percent'],50.)
        self.assertEqual(changes.loc['pe','percent'],-50.)
        self.assertTrue((changes.defined_datasets==len(DATASETS)).all())
        np.testing.assert_allclose(iteration_reductions(summary),-50.)

    def test_pc_bars_omit_pcm_dpc_and_missing_scores_stay_missing(self):
        rows=[{'dataset':d,'method':m,'pc_mean':np.nan if m in ('PCM','DPC','k-means++') else .7}
              for d in DATASETS for m in PRIMARY]
        values=metric_bars(pd.DataFrame(rows),'pc',PRIMARY)
        self.assertEqual(len(values.columns),len(PRIMARY)-3)
        for absent in ('PCM','DPC','k-means++'):self.assertNotIn(absent,values.columns)
        self.assertIn('LC-SoftFixedT',values.columns)
        values.loc['Wine','FCM']=np.nan
        fig,ax=plt.subplots()
        try:
            grouped_bars(ax,values,'PC',source={'kind':'summary','field':'pc_mean','scale':1.})
            series=capture_plot_data(fig)['axes'][0]['series']
            fcm=next(item for item in series if item['label']=='FCM')
            self.assertIsNone(fcm['values'][1])
            self.assertEqual(fcm['values'][0],.7)
        finally:plt.close(fig)

    def test_bar_rows_cover_every_dataset_once_and_record_their_own_datasets(self):
        rows_datasets=[d for _,names in DATASET_ROWS for d in names]
        self.assertEqual(sorted(rows_datasets),sorted(DATASETS))
        self.assertLessEqual(max(len(names) for _,names in DATASET_ROWS),8)
        values=pd.DataFrame({m:np.arange(len(DATASETS))+i for i,m in enumerate(PRIMARY)},index=DATASETS)
        fig=standalone_bar_figure(values,'test',source={'kind':'summary','field':'ari_mean','scale':1.})
        try:
            payload=capture_plot_data(fig)
            self.assertEqual([a['datasets'] for a in payload['axes']],[names for _,names in DATASET_ROWS])
            for axis in payload['axes']:
                for series in axis['series']:
                    expected=values.loc[axis['datasets'],series['label']].tolist()
                    self.assertEqual(series['values'],expected)
        finally:
            plt.close(fig)

    def test_heatmap_winners_handle_ties_direction_and_missing_cells(self):
        matrix=np.array([[.5,  .2, np.nan],
                         [.9,  .2, np.nan],
                         [.9,  .1, np.nan]])
        self.assertEqual(heatmap_winners(matrix,'ari'),[[1,0],[2,0],[0,1],[1,1]])   # ties all count
        self.assertEqual(heatmap_winners(matrix,'pe'),[[0,0],[2,1]])                # PE: lowest wins

    def test_all_reference_plot_types_and_original_heatmaps_are_registered(self):
        self.assertEqual(len(FIGURE_NAMES),22)
        self.assertEqual(len(set(FIGURE_NAMES)),22)
        for name in ['metric_comparison','heatmap_comparison']:      # removed: too small when printed
            self.assertNotIn(name,FIGURE_NAMES)
        for required in ['PC','PE','Sil','Pur','NMI','ARI','iteration','time','convergence_and_runtime',
                         'win_rank_summary','per_iteration_cost_profile','mcfcm_fuzzifier_statistics',
                         'lc_fixed_half_t_pipeline','temperature_controls']:
            self.assertIn(required,FIGURE_NAMES)

    def test_pipeline_describes_current_entropy_updates_and_cap_flag(self):
        fig=pipeline_figure()
        try:
            labels='\n'.join(t.get_text() for ax in fig.axes for t in ax.texts)
            self.assertIn('First-power',labels)
            self.assertIn('Keep t fixed',labels)
            self.assertIn('convergence flag',labels)
            self.assertNotIn('sigmoid',labels.lower())
            self.assertIn('nonconvergence','\n'.join(t.get_text() for t in fig.texts))
        finally:plt.close(fig)
