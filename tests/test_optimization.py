"""Optimized kernels against generic evolution, precision and cache regressions."""
import unittest
from unittest.mock import patch
import numpy as np
from nexqus import Circuit, Noise, NoiseModel, Simulator
from nexqus.backend import Backend


class OptimizationTests(unittest.TestCase):
    def test_local_operator_views_precision_and_input_preservation(self):
        rng=np.random.default_rng(54)
        for precision,dtype,tolerance in [('single',np.complex64,2e-6),('double',np.complex128,2e-13)]:
            for shape in ((2,)*5,(3,)+(2,)*4):
                original=(rng.normal(size=shape)+1j*rng.normal(size=shape)).astype(dtype)
                # Noncontiguous views exercise reshape/copy behavior as well.
                for state in (original,original.swapaxes(-1,-2)):
                    snapshot=state.copy()
                    for matrix in (np.array([[.6,-.8],[.8,.6]],dtype=dtype),
                                   np.diag(np.exp(1j*np.array([.2,-.7]))).astype(dtype),
                                   (rng.normal(size=(2,2))+1j*rng.normal(size=(2,2))).astype(dtype)):
                        batch=shape[0]==3
                        method='apply_batch' if batch else 'apply'
                        for wire in range(state.ndim-int(batch)):
                            actual=getattr(Backend(precision=precision),method)(state,matrix,(wire,))
                            expected=getattr(Backend(precision=precision,local_gates=False),method)(state,matrix,(wire,))
                            np.testing.assert_allclose(actual,expected,atol=tolerance,rtol=tolerance)
                            self.assertEqual(actual.dtype,dtype)
                    np.testing.assert_array_equal(state,snapshot)

    def test_density_fast_channels_against_generic_kraus(self):
        rng=np.random.default_rng(55)
        psi=rng.normal(size=8)+1j*rng.normal(size=8);psi/=np.linalg.norm(psi)
        channels=[factory(p) for factory in (Noise.amplitude_damping,Noise.phase_damping,
                                             Noise.bit_flip,Noise.phase_flip,Noise.depolarizing)
                  for p in (0,.17,1)]
        # A non-Pauli, noncanonical channel must retain its generic fallback.
        u=np.array([[1,1j],[1j,1]])/np.sqrt(2)
        channels.append(Noise(tuple(u@k@u.conj().T for k in Noise.amplitude_damping(.23).kraus),'depolarizing'))
        for precision,tolerance in [('single',3e-6),('double',3e-13)]:
            for wire in range(3):
                for ch in channels:
                    q=Circuit(3).h(2).cx(2,0).noise(ch,wire).ry(.31,1).cp(.7,2,0)
                    expected=Simulator(precision=precision,local_gates=False).run(
                        q,shots=0,initial_state=psi,noise_fastpath=False)
                    sim=Simulator(precision=precision)
                    for circuit in (q,sim.compile(q)):
                        actual=sim.run(circuit,shots=0,initial_state=psi)
                        np.testing.assert_allclose(actual.density_matrix,expected.density_matrix,atol=tolerance,rtol=0)
                        self.assertEqual(actual.density_matrix.dtype,np.complex64 if precision=='single' else np.complex128)
        np.testing.assert_allclose(np.linalg.norm(psi),1,atol=1e-14)

    def test_compilation_caches_value_based_channel_detection(self):
        q=Circuit(2).h(0).cx(0,1).noise(Noise.amplitude_damping(.2),1)
        model=NoiseModel(Noise.depolarizing(.1))
        sim=Simulator();compiled=sim.compile(q,model)
        expected=sim.run(q,noise=model,shots=0,noise_fastpath=False).density_matrix
        with patch.object(Noise,'pauli_weights',side_effect=AssertionError('Repeated classification')):
            with patch.object(Noise,'damping_spec',side_effect=AssertionError('Repeated classification')):
                actual=compiled.run(shots=0).density_matrix
        np.testing.assert_allclose(actual,expected,atol=2e-13)

    def test_exact_unseeded_zero_shot_run_does_not_initialize_rng(self):
        for method in ('statevector','density_matrix'):
            with patch('numpy.random.default_rng',side_effect=AssertionError('Unneeded RNG')):
                result=Simulator(method=method).run(Circuit(2).h(0).cx(0,1),shots=0)
                self.assertEqual(result.counts,{})
                self.assertAlmostEqual(result.expectation('ZZ'),1)
        with self.assertRaises(ValueError): Simulator().run(Circuit(1),shots=0,random_engine='unknown')
        with self.assertRaises((TypeError,ValueError)): Simulator().run(Circuit(1),shots=0,seed='invalid')
