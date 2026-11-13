"""Scalar neighbourhood and constrained-objective oracles for the controls."""
import unittest
import numpy as np
from scipy.optimize import minimize
from scipy.special import xlogy
from algo._local_entropy import LocalEntropyCMeans, local_coordinates, MARGINS
from benchmark.registry import make_model
from benchmark.config import PRIMARY


class FixedEntropyTests(unittest.TestCase):
    def test_local_mean_matches_scalar_neighbors_with_duplicates_and_ties(self):
        X = np.array([[0.],[0.],[1.],[2.],[8.],[9.],[10.]])
        points,inverse,counts = np.unique(X,axis=0,return_inverse=True,return_counts=True)
        k = min(len(points)-1,max(2,int(np.ceil(np.log2(len(points))))))
        rows = []
        for a,point in enumerate(points):
            neighbors = sorted([b for b in range(len(points)) if b!=a],key=lambda b:(np.sum((point-points[b])**2),b))[:k]
            average = (counts[a]*point+sum(counts[b]*points[b] for b in neighbors))/(counts[a]+sum(counts[b] for b in neighbors))
            rows.append((point+average)/2)
        Q = np.asarray(rows)[inverse]
        np.testing.assert_allclose(local_coordinates(X),Q-Q.mean(0),atol=1e-14)
        order = np.array([5,1,4,2,6,0,3])
        np.testing.assert_allclose(local_coordinates(X[order]),local_coordinates(X)[order],atol=1e-14)

    def test_first_cycle_matches_independent_membership_minima(self):
        X = np.array([[0.,1.],[.5,1.4],[3.,2.],[4.,3.],[7.,6.]])
        Q = local_coordinates(X)
        initial = np.random.default_rng(42).random((len(X),2))
        initial /= initial.sum(1)[:,None]
        centers = np.array([sum(initial[i,k]*Q[i] for i in range(len(Q)))/sum(initial[:,k]) for k in range(2)])
        rank = np.linalg.matrix_rank(Q)
        anchor = np.mean(np.sum(Q**2,axis=1))/(rank+2)
        for method,factor in [('LC-FixedT',1.),('LC-FixedHalfT',.5),('LC-SoftFixedT',2.)]:
            model = LocalEntropyCMeans(2,method=method,max_iter=1)
            fit = model.fit(X)
            np.testing.assert_allclose(fit.centers,centers,atol=1e-13)
            np.testing.assert_allclose(model.temperatures_,anchor*factor,atol=1e-13)
            distances = ((Q[:,None,:]-centers)**2).sum(2)
            for i,costs in enumerate(distances):
                def objective(u):return u@costs+anchor*factor*np.sum(xlogy(u,u))
                reference = minimize(objective,np.full(2,.5),bounds=[(1e-12,1)]*2,
                                     constraints={'type':'eq','fun':lambda u:u.sum()-1},
                                     method='SLSQP',options={'ftol':1e-12,'maxiter':1000})
                self.assertTrue(reference.success)
                self.assertLessEqual(abs(objective(fit.U[i])-reference.fun),1e-9)

    def test_descent_determinism_and_native_temperature(self):
        rng = np.random.default_rng(7)
        X = np.r_[rng.normal(-2,.4,(20,3)),rng.normal(2,.4,(20,3))]
        for method in MARGINS:
            model = LocalEntropyCMeans(2,method=method)
            fit = model.fit(X)
            self.assertTrue(fit.converged)
            self.assertTrue(np.all(np.diff(model.objective_history_)<=1e-9))
            np.testing.assert_allclose(fit.U.sum(1),1.,atol=1e-13)
            np.testing.assert_array_equal(fit.U,LocalEntropyCMeans(2,method=method).fit(X).U)
            np.testing.assert_array_equal(model.temperatures_,np.full(len(X),model.temperatures_[0]))

    def test_degenerate_inputs_and_retired_settings(self):
        for method in MARGINS:
            fit = LocalEntropyCMeans(3,method=method).fit(np.ones((5,2)))
            np.testing.assert_allclose(fit.U,1/3)
        with self.assertRaises(ValueError):LocalEntropyCMeans(2,method='LC-CMeans')
        with self.assertRaises(ValueError):LocalEntropyCMeans(2,tol=np.nan).fit(np.ones((5,2)))
        with self.assertRaises(ValueError):LocalEntropyCMeans(2).fit(np.array([[np.nan],[1.]]))
        with self.assertRaises(ValueError):LocalEntropyCMeans(3).fit(np.zeros((2,1)))

    def test_registry_constructs_only_the_eleven_benchmark_methods(self):
        self.assertEqual(len(PRIMARY),10)
        modules = ['fcm','pcm','pfcm','dpc','efcm','mc_fcm','lc_fixed_t','lc_fixed_half_t','lc_soft_fixed_t','kmeans_pp']
        for method,module in zip(PRIMARY,modules):
            self.assertEqual(type(make_model(method,2,42)).__module__, 'algo.'+module)
        for retired in ['DA-Stable-Transport','LC-CMeans','LC-CMeans-Soft','KMeans-Original','KMeans-Transport',
                        'KMeans-Spectral','KMeans-Spectral0','Author-WFCM-Spectral']:
            with self.assertRaises(ValueError):make_model(retired,2,42)
