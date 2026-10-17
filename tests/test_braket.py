"""No credentials and no cloud jobs: native local execution plus mocked AWS."""
from dataclasses import replace
from decimal import Decimal
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from nexqus import Circuit, Parameter, Noise
from nexqus.braket import prepare_braket, BraketBackend, BraketProgram, BraketResult


ARN = 'arn:aws:braket:us-east-1::device/qpu/example/test'
TASK = 'arn:aws:braket:us-east-1:123456789012:quantum-task/example-id'


def capabilities(gates=None, qubits=20, shots=(1, 10000)):
    return dict(action={'braket.ir.openqasm.program': {
        'supportedOperations': gates or ['i','h','x','y','z','rx','ry','rz',
                                        'phaseshift','cphaseshift','cnot','cz','swap']}},
        paradigm={'qubitCount':qubits}, service={'shotsRange':shots})


def native_result(counts, measured, shots):
    return SimpleNamespace(measurement_counts=counts, measured_qubits=measured,
                           task_metadata=SimpleNamespace(shots=shots))


class BraketOfflineTests(unittest.TestCase):
    def test_import_and_preparation_do_not_import_aws(self):
        code = """
import sys
from nexqus import Circuit
from nexqus.braket import prepare_braket, BraketBackend
p=prepare_braket(Circuit(8).rx(.5,0).cx(0,7))
b=BraketBackend('arn:aws:braket:us-east-1::device/qpu/example/test')
try: b.submit(p)
except PermissionError: pass
else: raise AssertionError('cloud submission must require opt-in')
assert not any(x=='braket' or x.startswith('braket.') or x=='boto3' for x in sys.modules)
"""
        subprocess.run([sys.executable, '-c', code], check=True, timeout=30)

    def test_translation_bind_inverse_snapshot_and_idle_wires(self):
        q = Circuit(8).rx(Parameter('theta'), 0).cx(0,7)
        p = prepare_braket(q, bindings={'theta':.5}, shots=123)
        q.x(3)
        self.assertEqual(p.instructions[:2], (('rx',(0,),(.5,)),('cnot',(0,7),())))
        self.assertEqual(p.gate_counts, {'rx':1,'cnot':1,'i':6})
        self.assertEqual(p.shots, 123)
        inverse = prepare_braket(Circuit(1).ry(.7,0).inverse())
        self.assertAlmostEqual(inverse.instructions[0][2][0], -.7)
        bound_inverse = prepare_braket(Circuit(1).rz(Parameter('a'),0).inverse(), bindings={'a':.6})
        self.assertAlmostEqual(bound_inverse.instructions[0][2][0], -.6)
        self.assertEqual(len(prepare_braket(Circuit(3)).instructions), 3)

    def test_names_and_display_parameters_cannot_change_execution(self):
        spoof = Circuit(1).unitary([[0,1],[1,0]],0,name='RX')
        spoof.operations[0] = replace(spoof.operations[0], params=(.7,))
        self.assertEqual(prepare_braket(spoof).instructions[0][0], 'x')
        arbitrary = Circuit(1).rx(.7,0)
        arbitrary.operations[0] = replace(arbitrary.operations[0], params=(.8,))
        with self.assertRaisesRegex(ValueError, 'decompose'):
            prepare_braket(arbitrary)

    def test_unsupported_and_bad_inputs_fail_before_import(self):
        bad = [Circuit(1).rx(Parameter('a'),0), Circuit(1).measure(0,'m'),
               Circuit(1).reset(0), Circuit(1).noise(Noise.depolarizing(.1),0),
               Circuit(2).permutation([0,2,1,3],[0,1]),
               Circuit(3).diagonal([1]*8,[0,1,2]),
               Circuit(1).measure(0,'m').x(0).c_if('m')]
        for q in bad:
            with self.subTest(q=q), self.assertRaises(ValueError): prepare_braket(q)
        for shots in [0,-1,True,1.2]:
            with self.assertRaises(ValueError): prepare_braket(Circuit(1),shots=shots)
        with self.assertRaises(TypeError): prepare_braket('not a circuit')
        for arn in ['', 'sv1', 'arn:other:device']:
            with self.assertRaises(ValueError): BraketBackend(arn)

    def test_offline_capability_and_cost_preflight(self):
        p=prepare_braket(Circuit(3).h(0).cx(0,2),shots=100)
        self.assertIs(p.validate_device(capabilities()),p)
        self.assertEqual(p.estimate_cost(task_price_usd='.30',shot_price_usd='.001'),Decimal('.400'))
        for bad in [None, {}, capabilities(qubits=2), capabilities(shots=(101,1000)),
                    capabilities(gates=['h','i'])]:
            with self.assertRaises(ValueError): p.validate_device(bad)
        for value in [None,True,-1,'NaN','Infinity','not a price']:
            with self.assertRaises(ValueError): p.estimate_cost(task_price_usd=value,shot_price_usd='.01')
        b=BraketBackend(ARN)
        with patch.object(b,'inspect_device',side_effect=AssertionError('network forbidden')):
            with self.assertRaises(PermissionError): b.submit(p)
            with self.assertRaises(ValueError):
                b.submit(p,allow_paid=True,task_price_usd='.3',shot_price_usd='.1',max_estimated_cost_usd=1)
            with self.assertRaises(ValueError): b.submit(p,allow_paid=True)
        cloud=BraketBackend('arn:aws:braket:::device/quantum-simulator/amazon/sv1')
        with self.assertRaisesRegex(ValueError,'time-based'):
            cloud.submit(p,allow_paid=True)

    def test_plan_validation_and_local_allocation_cap(self):
        for instructions in [(('unknown',(0,),()),), (('rx',(0,),()),),
                             (('rx',(0,),(float('nan'),)),), (('i',(1,),()),), ()]:
            with self.assertRaises(ValueError): BraketProgram(1,1,instructions)
        with self.assertRaises(ValueError): BraketProgram(1,0,(('i',(0,),()),))
        with self.assertRaisesRegex(ValueError,'max_qubits'):
            prepare_braket(Circuit(21)).run_local()
        mutable=[['rx',[0],[.5]]]
        p=BraketProgram(1,1,mutable); mutable[0][2][0]=1
        self.assertEqual(p.instructions,(('rx',(0,),(.5,)),))

    def test_result_ordering_sparse_probabilities_and_dense_bridge(self):
        r=BraketResult.from_native(native_result({'100':3,'011':1},[2,0,1],4),3,4)
        self.assertEqual(r.counts, {'001':3,'110':1})
        self.assertEqual(r.probabilities, {'001':.75,'110':.25})
        self.assertEqual(r.most_likely,'001')
        self.assertEqual(r.expectation('IIZ'),-.5)
        self.assertEqual(r.expectation('ZZI'),1)
        dense=r.to_result()
        np.testing.assert_allclose(dense.probabilities,[0,.75,0,0,0,0,.25,0])
        self.assertIsNone(dense.statevector)
        with self.assertRaises(ValueError): r.expectation('XXI')
        with self.assertRaises(ValueError): r.to_result(max_qubits=2)
        large=BraketResult.from_native(native_result({'0'*64:1},list(range(64)),1),64,1)
        self.assertEqual(len(large.probabilities),1)
        with self.assertRaises(ValueError): large.to_result()

    def test_malformed_results_do_not_become_plausible_measurements(self):
        for result in [native_result({'00':1},[0],1), native_result({'00':1},[0,0],1),
                       native_result({'02':1},[0,1],1),native_result({'00':True},[0,1],1),
                       native_result({'00':-1},[0,1],1),native_result({'00':.5},[0,1],1),
                       native_result({'00':1},[0,1],2),native_result({'00':2},[0,1],2)]:
            with self.assertRaises(ValueError): BraketResult.from_native(result,2,1)


if __name__=='__main__': unittest.main()
