import json
from pathlib import Path
import unittest
import numpy as np
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from scipy.optimize import minimize
from algo import _numerics as numerics
import metrics
from algo import fcm, pcm, pfcm, efcm
from benchmark import statistics
from algo.mc_fcm import MultipleCoefficientFuzzyCMeans

ROOT=Path(__file__).resolve().parents[1]

class ClusteringRegressionTests(unittest.TestCase):
    def setUp(self):
        self.X=np.array([[0.,0.],[.4,0.],[1.5,.5],[4.,1.],[4.2,1.1],[10.,4.]])

    def reference_values(self):
        return json.loads((ROOT/'tests/fixtures/mcfcm_paper_reference.json').read_text(encoding='utf-8'))

    def test_fuzzifiers_match_independent_paper_oracle(self):
        expected=self.reference_values()['fuzzifiers']
        actual,_=MultipleCoefficientFuzzyCMeans(2).fuzzifier_values(self.X)
        np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-12)

    def test_membership_and_center_updates_match_reference(self):
        reference=self.reference_values()
        m=np.asarray(reference['fuzzifiers'])
        d2=numerics.distances_to_centers_squared(self.X,np.asarray(reference['initial_centers']))
        actual=numerics.update_fcm_memberships_from_d2(d2,m)
        expected_U=np.asarray(reference['memberships'])
        np.testing.assert_allclose(actual,expected_U,atol=1e-12)
        expected_V=np.asarray(reference['updated_centers'])
        w=actual**m[:,None]
        np.testing.assert_allclose((w.T@self.X)/w.sum(0)[:,None],expected_V,atol=1e-12)

    def test_zero_distance_ties_and_known_m2_memberships(self):
        d2=np.array([[0.,0.,2.],[1.,4.,4.]])
        np.testing.assert_allclose(numerics.update_fcm_memberships_from_d2(d2,2.),
                                   [[.5,.5,0.],[2/3,1/6,1/6]],atol=1e-14)

    def test_invalid_fuzzifier_is_rejected(self):
        with self.assertRaises(ValueError):
            numerics.update_fcm_memberships_from_d2(np.array([[1.,2.]]),1.)


    def test_singleton_silhouette_matches_sklearn(self):
        X=np.array([[0.],[1.],[10.]])
        labels=[0,0,1]
        self.assertAlmostEqual(metrics.silhouette_score_np(X,labels),0.5962962962962962)
        self.assertAlmostEqual(metrics.silhouette_score_np(X,labels),silhouette_score(X,labels))

    def test_degenerate_silhouette_is_missing_not_zero(self):
        self.assertTrue(np.isnan(metrics.silhouette_score_np(self.X,np.zeros(len(self.X)))))

    def test_undefined_sample_sd_is_not_reported_as_zero(self):
        self.assertTrue(np.isnan(statistics.sample_std([np.nan,np.nan])))
        self.assertTrue(np.isnan(statistics.sample_std([.5])))
        self.assertEqual(statistics.mean_std_text(.5,np.nan),'0.5000 ± undefined')
        self.assertEqual(statistics.mean_std_text(np.nan,np.nan),'')

    def test_external_metrics_match_sklearn(self):
        cases=[([0,0,0],[0,0,0]),([0,1,2],[0,1,2]),([0,0,1,1,2,2],[1,0,1,1,0,2])]
        for truth,pred in cases:
            self.assertAlmostEqual(metrics.adjusted_rand_index(truth,pred),adjusted_rand_score(truth,pred))
            self.assertAlmostEqual(metrics.normalized_mutual_info(truth,pred),normalized_mutual_info_score(truth,pred))

    def test_pfcm_keeps_initial_scale_parameters(self):
        f=fcm.FuzzyCMeans(2,random_state=42).fit(self.X)
        expected=pfcm.PossibilisticFuzzyCMeans._eta(self.X,f.U,f.centers,2.)
        fit=pfcm.PossibilisticFuzzyCMeans(2,random_state=42).fit(self.X)
        self.assertAlmostEqual(fit.extra['eta_mean'],expected.mean(),places=12)

    def test_mc_fit_is_finite_normalized_and_monotone(self):
        fit=MultipleCoefficientFuzzyCMeans(2).fit(self.X)
        self.assertTrue(fit.converged)
        self.assertTrue(np.isfinite(fit.U).all())
        np.testing.assert_allclose(fit.U.sum(1),1.)
        self.assertEqual(fit.extra['objective_increases'],0)

    def test_fuzzy_memberships_solve_constrained_objective(self):
        # Independent numerical minimization, including heterogeneous exponents.
        for d2,m in [(np.array([1.,3.,8.]),1.4),(np.array([4.,1.,2.]),2.8)]:
            optimum=minimize(lambda u: np.dot(u**m,d2),np.full(3,1/3),
                             method='SLSQP',bounds=[(0,1)]*3,
                             constraints={'type':'eq','fun':lambda u:u.sum()-1},
                             options={'ftol':1e-13,'maxiter':500})
            self.assertTrue(optimum.success)
            calculated=numerics.update_fcm_memberships_from_d2(d2[None,:],m)[0]
            np.testing.assert_allclose(calculated,optimum.x,atol=2e-6)

    def test_entropy_memberships_solve_constrained_objective(self):
        d2=np.array([.2,1.3,2.1]); temperature=.7
        optimum=minimize(lambda u: np.dot(u,d2)+temperature*np.dot(u,np.log(u)),
                         np.full(3,1/3),method='SLSQP',bounds=[(1e-12,1)]*3,
                         constraints={'type':'eq','fun':lambda u:u.sum()-1},
                         options={'ftol':1e-13})
        self.assertTrue(optimum.success)
        actual=efcm.EntropyRegularizedFCM._soft_memberships(d2[None,:],temperature)[0]
        np.testing.assert_allclose(actual,optimum.x,atol=2e-6)

    def test_pcm_step_matches_fixed_eta_objective_updates(self):
        initial=fcm.FuzzyCMeans(2,max_iter=1,random_state=42).fit(self.X)
        weights=initial.U**2
        d2=((self.X[:,None,:]-initial.centers[None,:,:])**2).sum(2)
        eta=(weights*d2).sum(0)/weights.sum(0)
        typicalities=eta/(eta+d2)
        expected_centers=(typicalities**2).T@self.X/(typicalities**2).sum(0)[:,None]
        next_d2=((self.X[:,None,:]-expected_centers[None,:,:])**2).sum(2)
        next_t=eta/(eta+next_d2)
        fit=pcm.PossibilisticCMeans(2,max_iter=1,random_state=42).fit(self.X)
        np.testing.assert_allclose(fit.centers,expected_centers,atol=1e-12)
        np.testing.assert_allclose(fit.typicalities,typicalities,atol=1e-12)
        np.testing.assert_allclose(fit.U,typicalities,atol=1e-12)
        self.assertEqual(fit.membership_kind,'possibilistic')
        expected_objective=(typicalities**2*next_d2+eta*(1-typicalities)**2).sum()
        self.assertAlmostEqual(fit.objective,expected_objective,places=11)

    def test_pfcm_step_matches_joint_fuzzy_typicality_objective(self):
        initial=fcm.FuzzyCMeans(2,max_iter=1,random_state=42).fit(self.X)
        weights=initial.U**2
        d2=((self.X[:,None,:]-initial.centers[None,:,:])**2).sum(2)
        eta=(weights*d2).sum(0)/weights.sum(0)
        t=eta/(eta+d2)
        expected_centers=(weights+t**2).T@self.X/(weights+t**2).sum(0)[:,None]
        fit=pfcm.PossibilisticFuzzyCMeans(2,max_iter=1,random_state=42).fit(self.X)
        np.testing.assert_allclose(fit.centers,expected_centers,atol=1e-12)
        next_d2=((self.X[:,None,:]-expected_centers[None,:,:])**2).sum(2)
        self.assertAlmostEqual(fit.objective,((fit.U**2+t**2)*next_d2+eta*(1-t)**2).sum(),places=11)





    def test_mc_equal_density_guard_is_finite(self):
        X=np.zeros((6,2))
        fit=MultipleCoefficientFuzzyCMeans(2).fit(X)
        self.assertTrue(fit.converged)
        self.assertTrue(np.isfinite(fit.U).all())
        np.testing.assert_allclose(fit.U,.5)

if __name__=='__main__':
    unittest.main()
