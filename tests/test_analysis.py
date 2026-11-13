"""Complete-cell ranking and native fuzzy-metric eligibility checks."""
import unittest
import numpy as np
import pandas as pd
from benchmark.config import DATASETS
from benchmark.analysis import comparison,complete_means
from benchmark.statistics import metric_blocks,holm_adjust,wilcoxon_holm_tests
N=len(DATASETS)

class SimpleReportingTests(unittest.TestCase):
    def data(self):
        rows=[]
        for dataset in DATASETS:
            for seed in [42,43]:
                for name,value in [('FCM',.4),('New',.6),('DPC',.8)]:
                    rows.append({'dataset':dataset,'method':name,'seed':seed,'pc':value,'pe':1-value,
                                 'sil':value,'pur':value,'nmi':value,'ari':value,'total_time_s':.1})
        f=pd.DataFrame(rows)
        f.loc[f.method=='DPC',['pc','pe']]=np.nan
        return f

    def test_wins_count_only_label_metrics(self):
        rank,_=comparison(self.data(),['FCM','DPC','New'])
        rank=rank.set_index('method')
        self.assertEqual(rank.loc['DPC','eligible'],3*N)       # ARI, NMI, purity x every dataset
        self.assertEqual(rank.loc['DPC','wins'],3*N)
        self.assertTrue(np.isnan(rank.loc['DPC','pc_wins']))
        self.assertEqual(rank.loc['New','eligible'],3*N)
        self.assertEqual(rank.loc['New','pc_wins'],N)          # descriptive PC credit is kept ...
        self.assertEqual(rank.loc['New','wins'],0)             # ... but never counted as a win
        self.assertEqual(rank.loc['DPC','overall_score'],0.)
        self.assertEqual(rank.loc['FCM','overall_score'],1.)

    def test_missing_seed_cannot_become_a_complete_winning_cell(self):
        f=self.data()
        f.loc[(f.dataset=='Iris')&(f.method=='New')&(f.seed==42),'pc']=np.nan
        means=complete_means(f)
        self.assertTrue(np.isnan(means.loc[('Iris','New'),'pc']))
        rank,_=comparison(f,['FCM','DPC','New'])
        row=rank.set_index('method').loc['New']
        self.assertEqual(row.eligible,3*N)
        self.assertEqual(row.pc_wins,N-1)
        self.assertFalse(np.isnan(row.overall_score))       # label metrics are complete
        f.loc[(f.dataset=='Iris')&(f.method=='New')&(f.seed==42),'ari']=np.nan
        row=comparison(f,['FCM','DPC','New'])[0].set_index('method').loc['New']
        self.assertEqual(row.eligible,3*N-1)
        self.assertTrue(np.isnan(row.overall_score))

    def test_incomplete_method_is_dropped_from_tests_not_the_datasets(self):
        f=self.data()
        f.loc[(f.dataset=='Iris')&(f.method=='FCM'),'sil']=np.nan
        blocks=metric_blocks(f,'sil',['FCM','DPC','New'])
        self.assertEqual(list(blocks.columns),['DPC','New'])
        self.assertEqual(len(blocks),N)

    def test_holm_and_pairwise_wilcoxon_direction(self):
        np.testing.assert_allclose(holm_adjust([.01,.04,.03]),[.03,.06,.06])
        rows=wilcoxon_holm_tests(self.data(),['FCM','DPC','New'],metrics=('ari','pur'))
        pair=rows[(rows.metric=='ari')&(rows.method_a=='FCM')&(rows.method_b=='New')].iloc[0]
        self.assertEqual((pair.a_better,pair.b_better,pair.n_datasets),(0,N,N))
        self.assertTrue(pair.significant_holm_0_05)

