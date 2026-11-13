"""Source-equation oracles, native metric eligibility and interruption recovery."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
from benchmark.registry import run_single_method
from benchmark.statistics import friedman_nemenyi_tests
from algo.dpc import DensityPeaks, density_peak_quantities
from algo.fcm import FuzzyCMeans
from algo.pcm import PossibilisticCMeans
from algo.pfcm import PossibilisticFuzzyCMeans
from algo.efcm import EntropyRegularizedFCM
from algo.mc_fcm import MultipleCoefficientFuzzyCMeans
from benchmark.checkpoints import FitCheckpoint, WriterLock

ROOT = Path(__file__).resolve().parents[1]


class PaperProtocolTests(unittest.TestCase):
    def test_complete_mc_cycle_matches_independent_published_equation_oracle(self):
        reference = json.loads((ROOT/'tests/fixtures/mcfcm_paper_reference.json').read_text(encoding='utf-8'))
        fit = MultipleCoefficientFuzzyCMeans(2, max_iter=1, init_centers=reference['initial_centers']).fit(reference['X'])
        np.testing.assert_allclose(fit.U, reference['memberships'], atol=1e-12)
        np.testing.assert_allclose(fit.centers, reference['updated_centers'], atol=1e-12)
        self.assertAlmostEqual(fit.objective, 18.725873975003843, places=11)
        self.assertAlmostEqual(fit.extra['final_change'], 2.01302169504726, places=11)

    def test_pcm_pfcm_stop_on_normalized_center_change_and_keep_typicalities(self):
        X = np.array([[0.,0.],[1.,.2],[4.,2.],[5.,3.]])
        initial = FuzzyCMeans(2,max_iter=1,random_state=42).fit(X)
        for cls in [PossibilisticCMeans, PossibilisticFuzzyCMeans]:
            fit = cls(2,max_iter=1,tol=1e-5,random_state=42).fit(X)
            self.assertIsNotNone(fit.typicalities)
            change = np.linalg.norm(fit.centers-initial.centers)/fit.centers.size
            self.assertAlmostEqual(fit.extra['final_center_change'], change, places=12)
            self.assertEqual(fit.converged, change<=1e-5)

    def test_native_hard_and_possibilistic_outputs_have_no_fuzzy_partition_scores(self):
        X=np.array([[0.],[.1],[5.],[5.1]])
        dataset={'name':'toy','X':X,'y':np.array([0,0,1,1]),'n_clusters':2,'samples':4,'features':1,'classes':2}
        for method in ['PCM','DPC']:
            row=run_single_method(dataset,method,42)
            self.assertTrue(np.isnan(row['pc']))
            self.assertTrue(np.isnan(row['pe']))
            self.assertTrue(np.isfinite(row['ari']))
        self.assertTrue(np.isfinite(run_single_method(dataset,'PCM',42)['normalized_typicality_pc']))
        fit=DensityPeaks(2).fit(X)
        self.assertIsNone(fit.U)
        self.assertEqual(fit.membership_kind,'hard')
        self.assertEqual(len(np.unique(fit.labels)),2)

    def test_dpc_author_cutoff_and_ranked_assignment(self):
        X=np.array([[0.],[.1],[5.],[5.1]])
        distances,dc,rho,delta,order,nearest=density_peak_quantities(X)
        upper=np.sort(distances[np.triu_indices(4,1)])
        self.assertEqual(dc,upper[0])
        self.assertEqual(delta[order[0]],max(delta[order[1:]]))
        fit=DensityPeaks(2).fit(X)
        self.assertEqual(fit.labels[0],fit.labels[1])
        self.assertEqual(fit.labels[2],fit.labels[3])
        self.assertNotEqual(fit.labels[0],fit.labels[2])

    def test_entropy_temperature_is_explicit_and_not_dataset_median(self):
        X=np.array([[0.],[1.],[10.],[20.]])
        fit=EntropyRegularizedFCM(2,max_iter=1).fit(X)
        self.assertEqual(fit.extra['temperature'],1.)
        self.assertEqual(fit.extra['lambda_paper'],1.)

    def test_metric_statistics_exclude_inapplicable_methods_but_keep_external_scores(self):
        rows=[]
        for dataset in ['a','b','c']:
            for index,method in enumerate(['FCM','PCM','DPC','MC-FCM','LC-FixedHalfT']):
                rows.append({'dataset':dataset,'method':method,'pc': np.nan if method in ['PCM','DPC'] else .4+index*.1,
                             'ari': index*.1})
        tests,ranks,pairs=friedman_nemenyi_tests(pd.DataFrame(rows),
                ['FCM','PCM','DPC','MC-FCM','LC-FixedHalfT'],metrics=['pc','ari'])
        self.assertEqual(tests.set_index('metric').loc['pc','n_methods'],3)
        self.assertEqual(tests.set_index('metric').loc['ari','n_methods'],5)
        self.assertEqual(set(ranks[ranks.metric.eq('pc')].method),{'FCM','MC-FCM','LC-FixedHalfT'})



class RecoveryTests(unittest.TestCase):
    def test_abrupt_process_exit_preserves_only_committed_fits_and_resumes(self):
        with tempfile.TemporaryDirectory() as directory:
            environment=os.environ.copy()
            environment['PYTHONPATH']=str(ROOT/'src')
            script='''
import os,sys
from benchmark.checkpoints import FitCheckpoint
with FitCheckpoint(sys.argv[1], {'protocol':'fixture'}) as store:
    store.save({'dataset':'toy','configuration':'FCM','seed':42,'ari':0.5})
    store.connection.execute("INSERT INTO fits VALUES ('toy','FCM',43,'{}')")
    os._exit(17)
'''
            completed=subprocess.run([sys.executable,'-c',script,directory],env=environment,capture_output=True)
            self.assertEqual(completed.returncode,17,completed.stderr)
            with FitCheckpoint(directory,{'protocol':'fixture'}) as checkpoint:
                self.assertEqual([row['seed'] for row in checkpoint.rows()],[42])
                checkpoint.save({'dataset':'toy','configuration':'FCM','seed':43,'ari':0.6})
            with FitCheckpoint(directory,{'protocol':'fixture'}) as checkpoint:
                self.assertEqual(len(checkpoint.rows()),2)

    def test_changed_identity_and_concurrent_writer_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with FitCheckpoint(directory,{'protocol':'one'}):
                with self.assertRaises(RuntimeError):
                    with FitCheckpoint(directory,{'protocol':'one'}): pass
            with self.assertRaises(RuntimeError):
                with FitCheckpoint(directory,{'protocol':'two'}): pass


if __name__=='__main__':
    unittest.main()
