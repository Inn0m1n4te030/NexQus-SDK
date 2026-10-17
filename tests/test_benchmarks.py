"""Independent benchmark contracts; foreign SDKs remain optional dev tools."""
import importlib.util
from pathlib import Path
import sys
import unittest
import numpy as np

# The source-distribution test suite can run outside the checkout cwd. Adding
# its root (not src/) cannot shadow the installed nexqus runtime under test.
BENCHMARKS = Path(__file__).resolve().parents[1] / 'benchmarks'


def load(name):
    spec = importlib.util.spec_from_file_location('benchmark_'+name,BENCHMARKS/(name+'.py'))
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


reference = load('reference')
cases = load('cases')


class BenchmarkTests(unittest.TestCase):
    def test_reference_analytic_states_and_noise(self):
        spec = dict(n_qubits=2,method='statevector',operations=[dict(gate='RY',angle=np.pi,wires=[1]),
                    dict(gate='CX',wires=[1,0])])
        np.testing.assert_allclose(reference.dense_reference(spec),[0,0,0,1],atol=1e-14)
        spec = dict(n_qubits=1,method='density_matrix',operations=[dict(gate='RY',angle=np.pi,wires=[0]),
                    dict(gate='amplitude_damping',angle=.25,wires=[0])])
        np.testing.assert_allclose(reference.dense_reference(spec),np.diag([.25,.75]),atol=1e-14)
        spec['operations'] = [dict(gate='H',wires=[0]),dict(gate='depolarizing',angle=.2,wires=[0])]
        np.testing.assert_allclose(reference.dense_reference(spec),[[.5,.4],[.4,.5]],atol=1e-14)

    def test_accuracy_rejects_wrong_states_and_probabilities(self):
        psi = np.array([1,1j])/np.sqrt(2)
        self.assertTrue(reference.accuracy(psi,psi*1j,'statevector','double')['passed'])
        self.assertFalse(reference.accuracy(psi,psi[::-1],'statevector','double')['passed'])
        self.assertFalse(reference.accuracy(psi,psi*2,'statevector','double')['passed'])
        self.assertFalse(reference.accuracy(psi,psi,'statevector','double',np.array([1.,0.]))['passed'])
        rho = np.diag([.5,.5]).astype(complex); rho[0,1] = .1j
        self.assertFalse(reference.accuracy(np.eye(2)/2,rho,'density_matrix','double')['passed'])

    def test_all_small_specs_against_nexqus(self):
        # Load adapter's local reference import without changing the installed
        # runtime path or requiring foreign SDKs.
        previous = sys.modules.get('reference')
        sys.modules['reference'] = reference
        try: adapters = load('adapters')
        finally:
            if previous is None: del sys.modules['reference']
            else: sys.modules['reference'] = previous
        specs = cases.suite('correctness')
        self.assertEqual(len({spec['name'] for spec in specs}),len(specs))
        for spec in specs:
            expected = reference.dense_reference(spec)
            for precision in ('single','double'):
                with self.subTest(case=spec['name'],precision=precision):
                    run,_ = adapters.prepare('nexqus',spec,precision,1)
                    state, probabilities = run()
                    check = reference.accuracy(expected,state,spec['method'],precision,probabilities)
                    self.assertTrue(check['passed'],check)
