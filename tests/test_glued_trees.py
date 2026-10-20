"""Full random-cycle graph checks independent of the analytic column formulas."""
import json
import math
import unittest
import numpy as np
from nexqus.glued_trees import (GAMMA, ColumnQuantumWalk, column_model, glued_graph,
    bfs_hidden, classical_transport, classical_mean_hitting_time, verify_full_graph,
    compare_depths)
from nexqus.glued_trees_html import render_glued_trees


class GluedTreeTests(unittest.TestCase):
    def test_random_cycle_graph_has_correct_degrees_and_one_weld_cycle(self):
        for depth in (2,3,4):
            graph=glued_graph(depth); n=graph['vertices']; adjacency=graph['adjacency']
            self.assertEqual(n,2**(depth+2)-2)
            self.assertEqual(len(set(graph['names'])),n)
            self.assertTrue(all(0<=name<2**(2*depth) for name in graph['names']))
            for i,neighbors in enumerate(adjacency):
                self.assertEqual(len(neighbors),2 if i in (graph['entrance'],graph['exit']) else 3)
                self.assertEqual(len(set(neighbors)),len(neighbors)); self.assertNotIn(i,neighbors)
                for j in neighbors:
                    self.assertIn(i,adjacency[j]); self.assertEqual(abs(graph['columns'][i]-graph['columns'][j]),1)
            leaves={i for i,c in enumerate(graph['columns']) if c in (depth,depth+1)}
            visited=set(); pending=[next(iter(leaves))]
            while pending:
                i=pending.pop()
                if i in visited:continue
                visited.add(i)
                weld=[j for j in adjacency[i] if j in leaves]
                self.assertEqual(len(weld),2); pending.extend(j for j in weld if j not in visited)
            self.assertEqual(visited,leaves)
            self.assertEqual(graph,glued_graph(depth))

    def test_equitable_partition_and_seed_invariance_full_quantum_evolution(self):
        for depth in (2,3,4):
            for seed in (1,7,11):
                checked=verify_full_graph(depth,seed)
                self.assertLess(checked['invariance_max_error'],1e-13)
                self.assertLess(checked['state_max_error'],1e-12)
        counts,edges,h,k=column_model(4)
        self.assertEqual(counts.tolist(),[1,2,4,8,16,16,8,4,2,1])
        self.assertAlmostEqual(h[4,5],math.sqrt(2))
        self.assertAlmostEqual(h[0,1],1)
        np.testing.assert_allclose(k.sum(axis=0),0,atol=1e-15)

    def test_classical_column_evolution_matches_full_graph_generator(self):
        graph=glued_graph(3,7); n=graph['vertices']; a=np.zeros((n,n))
        for i,neighbors in enumerate(graph['adjacency']): a[neighbors,i]=1
        generator=GAMMA*(a-np.diag(a.sum(axis=0)))
        eigen,basis=np.linalg.eigh(generator)
        times=np.array([0.,.2,1.,4.,12.])
        full=(np.exp(times[:,None]*eigen)*basis[graph['entrance']])@basis.T
        aggregated=np.array([full[:,np.array(graph['columns'])==j].sum(axis=1) for j in range(8)]).T
        _,_,_,k=column_model(3); actual,average=classical_transport(k,times,12)
        np.testing.assert_allclose(actual,aggregated,atol=1e-13)
        np.testing.assert_allclose(actual.sum(axis=1),1,atol=1e-13)
        self.assertAlmostEqual(float(average.sum()),1)
        self.assertTrue(np.all(actual>=0))

    def test_analytic_uniform_time_metrics_match_numerical_quadrature(self):
        model=ColumnQuantumWalk(2); horizon=12.; times=np.linspace(0,horizon,12001)
        states=(np.exp(-1j*times[:,None]*model.values)*model.vectors[0])@model.vectors.T
        dt=horizon/(len(times)-1); y=abs(states[:,-1])**2
        numerical=float((y[0]/2+y[-1]/2+y[1:-1].sum())*dt/horizon)
        self.assertAlmostEqual(model.uniform_time_exit_probability(horizon),numerical,places=7)
        probabilities,average=classical_transport(model.k,times,horizon)
        y=probabilities[:,-1]
        numerical=float((y[0]/2+y[-1]/2+y[1:-1].sum())*dt/horizon)
        self.assertAlmostEqual(float(average[-1]),numerical,places=8)

    def test_classical_first_passage_formula_matches_full_absorbing_linear_system(self):
        graph=glued_graph(3); n=graph['vertices']; a=np.zeros((n,n))
        for i,neighbors in enumerate(graph['adjacency']): a[neighbors,i]=1
        k=GAMMA*(a-np.diag(a.sum(axis=0)))
        transient=[i for i in range(n) if i!=graph['exit']]
        first_passage=np.linalg.solve(k[np.ix_(transient,transient)].T,-np.ones(n-1))
        counts,edges,_,_=column_model(3)
        self.assertAlmostEqual(classical_mean_hitting_time(counts,edges),first_passage[0])

    def test_hidden_bfs_does_not_receive_exit_name_or_columns(self):
        graph=glued_graph(4)
        public={k:graph[k] for k in ('names','adjacency','entrance')}
        result=bfs_hidden(public)
        self.assertEqual(result['exit_name'],graph['names'][graph['exit']])
        self.assertEqual(result['neighbor_queries'],graph['vertices'])

    def test_sdk_normalization_padding_and_tiny_classical_exit(self):
        model=ColumnQuantumWalk(16)
        states=model.states([0,2,68]); np.testing.assert_allclose((abs(states)**2).sum(axis=1),1,atol=1e-13)
        counts,edges,h,k=column_model(32)
        probabilities,average=classical_transport(k,[0,66,132],132)
        self.assertTrue(np.all(probabilities>=0)); self.assertGreater(float(average[-1]),1e-12)
        self.assertLess(float(average[-1]),1e-9)
        self.assertAlmostEqual(float(average.sum()),1)

    def test_inputs_report_samples_and_script_escaping(self):
        for depth in (1,33,True,2.5):
            with self.assertRaises(ValueError): ColumnQuantumWalk(depth)
        model=ColumnQuantumWalk(2)
        for value in (-1,float('nan'),True,1+2j):
            with self.assertRaises(ValueError):model.circuit(value)
        for depths in ([],[2,2],[33]):
            with self.assertRaises(ValueError):compare_depths(depths)
        report=compare_depths([2],samples=17)
        row=report['results'][0]
        self.assertEqual(row['horizon'],12)
        for key,values in row['timing']['samples_ms'].items():
            self.assertEqual(len(values),5)
            self.assertTrue(all(np.isfinite(v) and v>=0 for v in values))
            self.assertEqual(row['timing']['median_ms'][key],sorted(values)[2])
        json.dumps(report,allow_nan=False)
        hostile='</script><img src=x onerror=alert(1)>'
        report['test']=hostile
        self.assertNotIn(hostile,render_glued_trees(report)); self.assertIn('\\u003c/script',render_glued_trees(report))


if __name__=='__main__': unittest.main()
