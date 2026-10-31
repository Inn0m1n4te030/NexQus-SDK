"""Analytic and finite-difference checks independent of report derivatives."""
import json
import unittest
import numpy as np
from nexqus import Circuit, Parameter, Simulator


class PredictionExplanationTests(unittest.TestCase):
    def test_shared_parameter_chain_rule_and_encoded_roles(self):
        x=Parameter('x'); w=Parameter('w')
        q=Circuit(1).ry(2*x,0).ry(w,0).ry(x,0)
        bindings={'x':.3,'w':-.2}
        d=q.explain(bindings=bindings,parameter_roles={'x':'feature','w':'weight'},sensitivities=True).to_dict()
        g={p['name']:p for p in d['sensitivity']['parameters']}
        self.assertAlmostEqual(d['sensitivity']['output'],np.cos(.7))
        self.assertAlmostEqual(g['x']['derivative'],-3*np.sin(.7))
        self.assertAlmostEqual(g['w']['derivative'],-np.sin(.7))
        self.assertEqual(g['x']['role'],'feature')
        self.assertEqual(bindings,{'x':.3,'w':-.2})
        self.assertEqual(q.parameters,('w','x'))

    def test_general_circuit_matches_finite_differences_with_custom_state(self):
        x=Parameter('x'); w=Parameter('w')
        q=Circuit(2).rx(x*w,0).ry(x,1).cx(1,0).rz(w,0)
        state=np.array([1,1j,2,-1],complex)/np.sqrt(7)
        values={'x':.4,'w':.7};sim=Simulator()
        d=q.explain(bindings=values,observable='XY',initial_state=state,sensitivities=True).to_dict()
        for p in d['sensitivity']['parameters']:
            plus=dict(values);minus=dict(values);plus[p['name']]+=1e-6;minus[p['name']]-=1e-6
            f=lambda v:sim.run(q.bind(v),shots=0,initial_state=state).expectation('XY')
            self.assertAlmostEqual(p['derivative'],(f(plus)-f(minus))/2e-6,places=7)

    def test_block_removal_matches_independent_circuit_and_preserves_stages(self):
        q=Circuit(2).ry(.7,0).cx(0,1).rz(.3,1)
        d=q.explain(blocks=[{'name':'layer','start':1,'stop':3}],interventions=[{'block':0,'skip':True}]).to_dict()
        variant=d['alternatives'][0]['trace']
        np.testing.assert_allclose(variant['frames'][-1]['probabilities'],Simulator().run(Circuit(2).ry(.7,0),shots=0).probabilities,atol=1e-12)
        self.assertEqual(len(variant['frames']),len(d['frames']))
        self.assertEqual(len(q.operations),3)
        self.assertIn('layer',d['alternatives'][0]['label'])

    def test_validation_opt_in_empty_and_export_safety(self):
        q=Circuit(1).h(0)
        self.assertNotIn('sensitivity',q.explain().to_dict())
        self.assertEqual(q.explain(sensitivities=True).to_dict()['sensitivity']['parameters'],[])
        for request in [{'block':True,'skip':True},{'block':0,'skip':True},{'block':0,'skip':False}]:
            with self.assertRaises(ValueError):q.explain(interventions=[request])
        with self.assertRaises(ValueError):q.explain(sensitivities='yes')
        x=Parameter('</script><b>')
        report=Circuit(1).ry(x,0).explain(bindings={'</script><b>':.2},sensitivities=True)
        self.assertNotIn('</script><b>',report.to_html())
        json.dumps(report.to_dict(),allow_nan=False)

    def test_stationary_point_and_resource_bound(self):
        x=Parameter('x')
        self.assertAlmostEqual(Circuit(1).ry(x,0).explain(bindings={'x':0},sensitivities=True).to_dict()['sensitivity']['parameters'][0]['derivative'],0)
        q=Circuit(1)
        for i in range(33):q.ry(Parameter(f'p{i}'),0)
        with self.assertRaisesRegex(ValueError,'32 symbolic'):
            q.explain(bindings={f'p{i}':.1 for i in range(33)},sensitivities=True)
