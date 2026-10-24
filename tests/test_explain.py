"""Analytic ground truths, trace invariants, boundedness and export safety."""
import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from nexqus import Circuit, Simulator, Parameter, Noise, algorithms


class ExplainTests(unittest.TestCase):
    def test_interventions_match_independent_circuits_and_preserve_source(self):
        theta=Parameter('theta')
        circuit=Circuit(2).ry(theta,0).cx(0,1)
        data=circuit.explain(bindings={'theta':.5},interventions=[
            {'operation':0,'angle':1.2},{'operation':1,'skip':True}]).to_dict()
        expected=[Circuit(2).ry(1.2,0).cx(0,1),Circuit(2).ry(.5,0)]
        for variant,reference in zip(data['alternatives'],expected):
            frame=variant['trace']['frames'][-1]
            result=Simulator().run(reference,shots=0)
            np.testing.assert_allclose(frame['probabilities'],result.probabilities,atol=1e-12)
            self.assertAlmostEqual(frame['expectation'],result.expectation('ZI'))
        self.assertEqual(circuit.parameters,('theta',))
        self.assertEqual(len(circuit.operations),2)
        self.assertAlmostEqual(data['alternatives'][1]['trace']['frames'][-1]['qubits'][0]['entropy'],0)

    def test_intervention_validation_and_aggregate_budget(self):
        q=Circuit(1).rx(.5,0)
        for request in [{'operation':True,'skip':True},{'operation':1,'skip':True},
                        {'operation':0,'skip':False},{'operation':0,'angle':float('nan')},
                        {'operation':0,'skip':True,'angle':1}]:
            with self.assertRaises(ValueError): q.explain(interventions=[request])
        with self.assertRaises(ValueError): q.explain(interventions=[{'operation':0,'skip':True}]*17)
        with self.assertRaises(ValueError): Circuit(1).h(0).explain(interventions=[{'operation':0,'angle':1}])
        base=q.explain().to_dict()
        size=len(json.dumps(base,ensure_ascii=False,allow_nan=False).encode())
        with self.assertRaisesRegex(ValueError,'Combined'):
            q.explain(interventions=[{'operation':0,'skip':True}],max_report_mb=size*1.5/1024**2)

    def test_zero_angle_requires_explicit_verified_axis(self):
        q=Circuit(1).ry(0,0)
        with self.assertRaisesRegex(ValueError,'ambiguous'):
            q.explain(interventions=[{'operation':0,'angle':.7}])
        data=q.explain(interventions=[{'operation':0,'angle':.7,'axis':'ry'}]).to_dict()
        np.testing.assert_allclose(data['alternatives'][0]['trace']['frames'][-1]['probabilities'],
                                   [np.cos(.35)**2,np.sin(.35)**2],atol=1e-12)
        with self.assertRaises(ValueError):
            Circuit(1).rx(.5,0).explain(interventions=[{'operation':0,'angle':.7,'axis':'ry'}])

    def test_bell_explains_joint_state_and_reduced_mixedness(self):
        data=Circuit(2).h(0).cx(0,1).explain(observable='ZZ').to_dict()
        self.assertEqual(len(data['frames']),3)
        first,final=data['frames'][1:]
        np.testing.assert_allclose(final['probabilities'],[.5,0,0,.5],atol=1e-12)
        self.assertAlmostEqual(final['expectation'],1)
        self.assertAlmostEqual(final['z_covariance'][0][1],1)
        for q in final['qubits']:
            np.testing.assert_allclose(q['bloch'],[0,0,0],atol=1e-12)
            self.assertAlmostEqual(q['purity'],.5); self.assertAlmostEqual(q['entropy'],1)
        self.assertAlmostEqual(first['qubits'][0]['purity'],1)
        self.assertIn('entangled',final['explanation'])

    def test_interference_and_global_phase_are_distinguished(self):
        d=Circuit(1).h(0).z(0).h(0).explain().to_dict()
        phase=d['frames'][2]; end=d['frames'][3]
        self.assertAlmostEqual(phase['probability_change'],0)
        self.assertAlmostEqual(phase['phase_change'],np.pi)
        self.assertAlmostEqual(phase['fidelity_to_previous'],0)
        self.assertIn('Relative phases changed',phase['explanation'])
        np.testing.assert_allclose(end['probabilities'],[0,1],atol=1e-12)
        global_phase=Circuit(1).h(0).unitary(1j*np.eye(2),0)
        f=global_phase.explain().to_dict()['frames'][-1]
        self.assertAlmostEqual(f['phase_change'],0)
        self.assertAlmostEqual(f['fidelity_to_previous'],1)

    def test_bloch_y_sign_and_pure_state_entanglement_scope(self):
        f=Circuit(1).h(0).phase(np.pi/2,0).explain().to_dict()['frames'][-1]
        np.testing.assert_allclose(f['qubits'][0]['bloch'],[0,1,0],atol=1e-12)
        self.assertAlmostEqual(f['qubits'][0]['entropy'],0)
        ghz=algorithms.ghz(3).explain().to_dict()['frames'][-1]
        self.assertAlmostEqual(ghz['qubits'][0]['entropy'],1)
        # Pair covariance alone is not labelled as pair entanglement.
        self.assertIn('remaining qubits',ghz['explanation'])

    def test_every_checkpoint_matches_existing_simulation(self):
        q=Circuit(3).rx(.4,2).ry(-.3,0).cp(.7,2,1).swap(0,2)
        q.permutation([1,2,0,3],[2,0]).compose(Circuit(3).h(1).rz(.2,2).inverse())
        data=q.explain(observable='XYZ').to_dict()
        prefix=Circuit(3)
        for i,frame in enumerate(data['frames']):
            if i: prefix.operations.append(q.operations[i-1])
            expected=Simulator().run(prefix,shots=0)
            actual=np.array(frame['amplitudes'])@np.array([1,1j])
            np.testing.assert_allclose(actual,expected.statevector,atol=1e-12)
            self.assertAlmostEqual(frame['expectation'],expected.expectation('XYZ'))

    def test_parameters_annotations_and_source_isolation(self):
        q=Circuit(2).ry(Parameter('x'),0).rx(Parameter('w'),1).cx(0,1)
        report=q.explain(bindings={'x':.3,'w':.7},parameter_roles={'x':'feature','w':'weight'},
            blocks=[dict(name='Encode',start=0,stop=1,role='encoding')])
        q.x(0)
        a=report.to_dict(); self.assertEqual(len(a['operations']),3)
        self.assertEqual(a['operations'][0]['parameters'],['x'])
        self.assertEqual(a['operations'][1]['parameters'],['w'])
        self.assertEqual(a['operations'][0]['block'],0)
        self.assertIsNone(a['operations'][1]['block'])
        a['frames'][0]['probabilities'][0]=99
        self.assertEqual(report.to_dict()['frames'][0]['probabilities'][0],1)

    def test_names_do_not_invent_gate_semantics(self):
        q=Circuit(1).unitary([[0,1],[1,0]],0,name='H')
        d=q.explain().to_dict()
        self.assertIn('X exchanges',d['operations'][0]['action'])
        self.assertNotIn('Hadamard',d['operations'][0]['action'])

    def test_limits_and_unsupported_cases_fail_before_simulation(self):
        for q,options in [(Circuit(11),{}),(Circuit(2).h(0).x(0),{'max_steps':1}),
                          (Circuit(8),{'max_report_mb':.001}),
                          (Circuit(1).noise(Noise.depolarizing(.1),0),{}),
                          (Circuit(1).measure(0,'m'),{}),(Circuit(1).reset(0),{})]:
            with patch('nexqus.explain.Simulator',side_effect=AssertionError('allocated state')):
                with self.assertRaises(ValueError): q.explain(**options)
        for options in [dict(max_qubits=True),dict(max_steps=0),dict(max_report_mb=float('nan')),
                        dict(max_report_mb=0),dict(observable='XX'),dict(title=''),
                        dict(parameter_roles={'unknown':'feature'}),
                        dict(blocks=[dict(name='bad',start=0,stop=2)])]:
            with self.assertRaises(ValueError): Circuit(1).h(0).explain(**options)
        with self.assertRaises(ValueError): Circuit(1).ry(Parameter('x'),0).explain()
        with self.assertRaises(ValueError): Circuit(1).explain(initial_state=[1,1])
        with self.assertRaises(ValueError):
            Circuit(1).h(0).x(0).explain(blocks=[dict(name='a',start=0,stop=2),dict(name='b',start=1,stop=2)])

    def test_empty_custom_initial_state_and_export_are_valid(self):
        report=Circuit(1).explain(initial_state=[0,1]); data=report.to_dict()
        self.assertEqual(len(data['frames']),1)
        self.assertEqual(data['frames'][0]['expectation'],-1)
        self.assertIn('supplied',data['frames'][0]['explanation'])
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'report.html'; output=report.to_html(filename=p)
            self.assertEqual(str(output),p.read_text())
            self.assertIn('sandbox="allow-scripts allow-downloads"',output._repr_html_())

    def test_untrusted_labels_are_json_data_and_round_trip_without_markup(self):
        hostile='</script><img src=x onerror="alert(1)"> & text'
        report=Circuit(1).unitary(np.eye(2),0,name=hostile).explain(title=hostile)
        output=report.to_html()
        self.assertNotIn(hostile,output)
        payload=re.search(r'<script type="application/json" id="trace-data">(.*?)</script>',output,re.S).group(1)
        self.assertEqual(json.loads(payload)['title'],hostile)
        self.assertNotIn('<img',payload)
        self.assertNotIn('https://',output)


if __name__=='__main__': unittest.main()
