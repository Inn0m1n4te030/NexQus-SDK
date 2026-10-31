"""Independent series/checkpoint truth and optional device parity."""
import unittest
from unittest.mock import patch
import numpy as np
from nexqus import Circuit, Parameter, Simulator


class ExplanationSeriesTests(unittest.TestCase):
    def setUp(self):
        self.x=Parameter('x');self.w=Parameter('w')
        self.q=Circuit(1).ry(self.x,0).ry(self.w,0)
        self.options=dict(bindings={'x':.3,'w':.4},sensitivities=True,
            comparisons=[dict(name='B',bindings={'x':1.1,'w':.4})],
            training_history=[dict(epoch=0,bindings={'x':.3,'w':-.2},loss=.2),
                              dict(epoch=4,bindings={'x':.3,'w':.1})])

    def test_all_frames_and_gradients_have_independent_analytic_truth(self):
        d=self.q.explain(**self.options).to_dict()
        traces=[d,d['comparisons'][0]['trace']]+[r['trace'] for r in d['training_history']]
        for trace in traces:
            vals={p['name']:p['value'] for p in trace['parameters']}
            self.assertAlmostEqual(trace['frames'][1]['expectation'],np.cos(vals['x']))
            self.assertAlmostEqual(trace['frames'][2]['expectation'],np.cos(vals['x']+vals['w']))
            for g in trace['sensitivity']['parameters']:
                self.assertAlmostEqual(g['derivative'],-np.sin(vals['x']+vals['w']))
        self.assertNotIn('loss',d['training_history'][1])
        self.assertEqual(d['training_history'][0]['loss'],.2)
        self.assertNotIn('comparisons',traces[-1])
        # Caller edits do not alter exported values.
        self.options['training_history'][0]['bindings']['w']=7
        self.assertAlmostEqual(traces[2]['parameters'][0]['value'],-.2)

    def test_matched_interventions_on_both_inputs(self):
        d=self.q.explain(**self.options,blocks=[dict(name='encoding',start=0,stop=1)],
                         interventions=[dict(block=0,skip=True)]).to_dict()
        for trace in [d,d['comparisons'][0]['trace']]:
            self.assertAlmostEqual(trace['alternatives'][0]['trace']['frames'][-1]['expectation'],np.cos(.4))
        self.assertEqual(d['training_history'][0]['trace']['alternatives'],[])

    def test_bad_records_and_budget_reject_before_simulation(self):
        bad=[dict(training_history=[dict(epoch=2**53,bindings={'x':.3,'w':.4})]),
             dict(training_history=[dict(epoch=i,bindings={'x':.3,'w':.4}) for i in range(33)]),
             dict(comparisons=[dict(name='',bindings={'x':.3,'w':.4})]),
             dict(comparisons=[dict(name='B',bindings={'x':.3})]),
             dict(training_history=[dict(epoch=True,bindings={'x':.3,'w':.4})]),
             dict(training_history=[dict(epoch=0,bindings={'x':.3,'w':.4},loss=float('nan'))]),
             dict(training_history=[dict(epoch=0,bindings={'x':.3,'w':.4})]*2),
             dict(comparisons=[dict(name='B',bindings={'x':.3,'w':.4})]*9),
             dict(comparisons=[dict(name='B',bindings={'x':.3,'w':.4})]*8,max_report_mb=.01)]
        for options in bad:
            with self.subTest(options=options),patch.object(Simulator,'run',side_effect=AssertionError('allocated')):
                with self.assertRaises(ValueError):self.q.explain(bindings={'x':.3,'w':.4},**options)

    def test_custom_initial_state_retained_in_all_series(self):
        d=self.q.explain(**self.options,initial_state=[0,1]).to_dict()
        for trace in [d,d['comparisons'][0]['trace'],d['training_history'][0]['trace']]:
            v={p['name']:p['value'] for p in trace['parameters']}
            self.assertAlmostEqual(trace['frames'][-1]['expectation'],-np.cos(v['x']+v['w']))
            self.assertAlmostEqual(trace['sensitivity']['parameters'][0]['derivative'],np.sin(v['x']+v['w']))

    def check_device(self,device):
        # Shared nonlinear expressions plus layer removals exercise PR 26 as well.
        q=Circuit(2).rx(self.x*self.w,0).ry(self.x,1).cx(1,0).rz(self.w,0)
        args=dict(self.options,observable='XY',blocks=[dict(name='layer',start=2,stop=4)],
                  interventions=[dict(block=0,skip=True)],initial_state=np.array([1,1j,2,-1])/np.sqrt(7))
        cpu=q.explain(**args).to_dict();other=q.explain(**args,device=device,backend='torch').to_dict()
        pairs=[(cpu,other),(cpu['comparisons'][0]['trace'],other['comparisons'][0]['trace'])]
        pairs+=list(zip([r['trace'] for r in cpu['training_history']],[r['trace'] for r in other['training_history']]))
        pairs.append((cpu['alternatives'][0]['trace'],other['alternatives'][0]['trace']))
        for a,b in pairs:
            for fa,fb in zip(a['frames'],b['frames']):
                np.testing.assert_allclose(fa['probabilities'],fb['probabilities'],atol=1e-11)
                self.assertAlmostEqual(fa['expectation'],fb['expectation'],places=10)
            if 'sensitivity' in a:
                np.testing.assert_allclose([p['derivative'] for p in a['sensitivity']['parameters']],
                                           [p['derivative'] for p in b['sensitivity']['parameters']],atol=1e-10)
        self.assertEqual(other['execution']['device'],device)
        self.assertEqual(other['execution']['backend'],'torch')

    def test_torch_cpu_parity(self):
        try:import torch
        except ImportError:self.skipTest('Optional PyTorch not installed')
        self.check_device('cpu')

    def test_cuda_parity(self):
        try:import torch
        except ImportError:self.skipTest('Optional PyTorch not installed')
        if not torch.cuda.is_available():self.skipTest('CUDA GPU unavailable')
        self.check_device('cuda:0')
