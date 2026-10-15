"""Dynamic protocol and quantum/classical correlation regressions."""
import unittest
import numpy as np
from nexqus import Circuit,Simulator,Parameter,QuantumFunction,Noise,NoiseModel
from nexqus.validation import compare_results


def teleport_circuit(corrections=True):
    q=Circuit(3).h(1).cx(1,2).cx(0,1).h(0).measure(0,'z').measure(1,'x')
    if corrections: q.x(2).c_if('x',1).z(2).c_if('z',1)
    return q


class DynamicTests(unittest.TestCase):
    def test_projective_measurement_removes_coherence(self):
        q=Circuit(1).h(0).measure(0,'m').h(0)
        r=Simulator().run(q,shots=0)
        np.testing.assert_allclose(r.density_matrix,np.eye(2)/2,atol=1e-12)
        self.assertEqual(r.classical_keys,('m',))
        self.assertAlmostEqual(r.classical_probabilities['0'],.5)
        self.assertEqual(r.metadata['method'],'branch_density')

    def test_deterministic_measurement(self):
        r=Simulator().run(Circuit(1).x(0).measure(0,'m'),shots=30,seed=1)
        self.assertEqual(r.classical_probabilities,{'1':1.})
        self.assertEqual(r.classical_counts,{'1':30})
        self.assertEqual(r.joint_counts,{'1|1':30})
        self.assertEqual(r.metadata['branch_count'],1)

    def test_feed_forward_both_condition_values(self):
        for expected in (0,1):
            q=Circuit(1).h(0).measure(0,'m').x(0).c_if('m',1-expected)
            r=Simulator().run(q,shots=100,seed=3)
            self.assertEqual(r.counts,{str(expected):100})

    def test_joint_classical_quantum_correlations(self):
        q=Circuit(2).h(0).measure(0,'m').x(1).c_if('m',1)
        for method in ('auto','trajectory'):
            r=Simulator(method=method).run(q,shots=1000,seed=8,trajectories=512,batch_size=32)
            self.assertTrue(set(r.joint_counts)<=set(['0|00','1|11']))
            self.assertEqual(sum(r.joint_counts.values()),1000)
            self.assertEqual(r.classical_counts.get('0',0),r.counts.get('00',0))
            self.assertEqual(r.classical_counts.get('1',0),r.counts.get('11',0))

    def test_repeated_measurement_is_repeatable(self):
        q=Circuit(1).h(0).measure(0,'first').measure(0,'second')
        for method in ('auto','trajectory'):
            r=Simulator(method=method).run(q,shots=100,seed=4,trajectories=128,batch_size=13)
            self.assertTrue(set(r.classical_probabilities)<=set(['00','11']))
            self.assertTrue(set(r.classical_counts)<=set(['00','11']))

    def test_overwritten_bit_merges_branches(self):
        q=Circuit(1).h(0).measure(0,'m').h(0).measure(0,'m')
        r=Simulator().run(q,shots=0)
        self.assertEqual(r.metadata['branch_count'],2)
        np.testing.assert_allclose(r.probabilities,[.5,.5],atol=1e-12)
        self.assertEqual(r.classical_keys,('m',))

    def test_reset_entangled_qubit(self):
        r=Simulator().run(Circuit(2).h(0).cx(0,1).reset(0),shots=0)
        np.testing.assert_allclose(r.probabilities,[.5,.5,0,0],atol=1e-12)
        np.testing.assert_allclose(r.reduced_density([0]),[[1,0],[0,0]],atol=1e-12)
        np.testing.assert_allclose(r.reduced_density([1]),np.eye(2)/2,atol=1e-12)
        self.assertEqual(r.classical_probabilities,{})

    def test_conditional_reset(self):
        q=Circuit(2).h(0).cx(0,1).measure(0,'m').reset(1).c_if('m',1)
        r=Simulator().run(q,shots=0)
        np.testing.assert_allclose(r.probabilities,[.5,0,.5,0],atol=1e-12)

    def test_reset_trajectory(self):
        q=Circuit(2).h(0).cx(0,1).reset(0)
        r=Simulator(method='trajectory').run(q,trajectories=3000,batch_size=64,shots=100,seed=2)
        np.testing.assert_allclose(r.probabilities,[.5,.5,0,0],atol=.035)
        self.assertTrue(set(r.counts)<=set(['00','01']))

    def test_teleportation_random_complex_states(self):
        from nexqus import algorithms
        rng=np.random.default_rng(8)
        for _ in range(12):
            psi=rng.normal(size=2)+1j*rng.normal(size=2); psi/=np.linalg.norm(psi)
            initial=np.kron(psi,[1,0,0,0])
            r=Simulator().run(teleport_circuit(),initial_state=initial,shots=0)
            np.testing.assert_allclose(r.reduced_density([2]),np.outer(psi,psi.conj()),atol=1e-12)
            built_in=Simulator().run(algorithms.teleportation(),initial_state=initial,shots=0)
            np.testing.assert_allclose(built_in.reduced_density([2]),r.reduced_density([2]),atol=1e-12)
            np.testing.assert_allclose(list(r.classical_probabilities.values()),[.25]*4,atol=1e-12)

    def test_teleportation_corrections_are_necessary(self):
        initial=np.kron([1/np.sqrt(2),1j/np.sqrt(2)],[1,0,0,0])
        r=Simulator().run(teleport_circuit(False),initial_state=initial,shots=0)
        np.testing.assert_allclose(r.reduced_density([2]),np.eye(2)/2,atol=1e-12)

    def test_conditional_gate_noise_only_on_executed_branches(self):
        x=np.array([[0,1],[1,0]],complex)
        noise=NoiseModel().after('conditionalX',Noise.bit_flip(.2))
        for value,expected in ((0,[0,0,1,0]),(1,[0,0,.2,.8])):
            q=Circuit(2).x(0).measure(0,'m').unitary(x,1,'conditionalX').c_if('m',value)
            r=Simulator().run(q,noise=noise,shots=0)
            np.testing.assert_allclose(r.probabilities,expected,atol=1e-12)

    def test_dynamic_noisy_trajectory_converges(self):
        q=Circuit(2).h(0).measure(0,'m').ry(.8,1).c_if('m',1).reset(0)
        model=NoiseModel(Noise.amplitude_damping(.2),Noise.depolarizing(.1))
        exact=Simulator().run(q,noise=model,shots=0)
        sampled=Simulator(method='trajectory').run(q,noise=model,shots=0,trajectories=4000,batch_size=71,seed=9)
        np.testing.assert_allclose(sampled.probabilities,exact.probabilities,atol=.025)
        for key in exact.classical_probabilities:
            self.assertAlmostEqual(sampled.classical_probabilities.get(key,0),exact.classical_probabilities[key],delta=.025)

    def test_terminal_readout_does_not_flip_classical_control(self):
        q=Circuit(1).x(0).measure(0,'m')
        r=Simulator().run(q,readout_error=1,shots=100,seed=4)
        self.assertEqual(r.joint_counts,{'1|0':100})
        np.testing.assert_allclose(r.density_matrix,[[0,0],[0,1]])

    def test_dynamic_compiled_snapshot_and_parameter_bind(self):
        t=Parameter('t')
        q=Circuit(1).h(0).measure(0,'m').ry(t,0).c_if('m',1)
        bound=q.bind({'t':.3}); sim=Simulator()
        frozen=sim.compile(bound)
        a=sim.run(bound,seed=8); b=frozen.run(seed=8)
        compare_results(a,b)
        self.assertEqual(a.joint_counts,b.joint_counts)
        bound.x(0)
        compare_results(b,frozen.run(seed=8))

    def test_composition_classical_mapping(self):
        block=Circuit(1).h(0).measure(0,'m').x(0).c_if('m',1)
        q=Circuit(2).compose(block,[1],classical_map={'m':'right'})
        q.compose(block,[0],classical_map={'m':'left'})
        r=Simulator().run(q,shots=30,seed=1)
        self.assertEqual(r.classical_keys,('right','left'))
        self.assertEqual(r.counts,{'00':30})

    def test_sampling_reproducible_partial_batch(self):
        q=Circuit(1).h(0).measure(0,'m').reset(0)
        sim=Simulator(method='trajectory')
        kw=dict(trajectories=19,batch_size=8,shots=50,seed=2)
        a=sim.run(q,**kw); b=sim.run(q,**kw)
        np.testing.assert_array_equal(a.probabilities,b.probabilities)
        self.assertEqual(a.joint_counts,b.joint_counts)

    def test_reduced_density_wire_order(self):
        q=Circuit(3).x(0).h(1)
        r=Simulator().run(q,shots=0)
        np.testing.assert_allclose(r.reduced_density([2,0]),np.diag([0,1,0,0]),atol=1e-12)
        rho=Simulator(method='density_matrix').run(q,shots=0)
        np.testing.assert_allclose(rho.reduced_density([2,0]),r.reduced_density([2,0]),atol=1e-12)

    def test_classical_validation_errors(self):
        with self.assertRaises(ValueError): Circuit(1).x(0).c_if('missing')
        with self.assertRaises(ValueError): Circuit(1).measure(0,'')
        with self.assertRaises(ValueError): Circuit(1).measure(0,'m').c_if('m')
        with self.assertRaises(ValueError): Circuit(1).measure(0,'m').x(0).c_if('m',2)
        with self.assertRaises(ValueError): Circuit(1).measure(0,'m').inverse()
        with self.assertRaises(ValueError): QuantumFunction(Circuit(1).reset(0),'Z')
        with self.assertRaises(ValueError): Simulator(method='statevector').run(Circuit(1).measure(0,'m'))
        with self.assertRaises(ValueError): Circuit(1).reset(0).c_if('m')

    def test_branch_memory_guard(self):
        q=Circuit(4)
        for j in range(4): q.h(j).measure(j,str(j))
        with self.assertRaises(MemoryError): Simulator().run(q,max_branches=4)
        with self.assertRaises(MemoryError): Simulator(max_memory_mb=.0001).run(q)
        r=Simulator(method='trajectory').run(q,max_branches=4,trajectories=20,batch_size=4,shots=0)
        self.assertAlmostEqual(r.probabilities.sum(),1)


try:
    import torch
except ImportError:
    torch=None


@unittest.skipIf(torch is None,'Optional PyTorch not installed')
class TorchV05Tests(unittest.TestCase):
    def compare(self,device):
        initial=np.kron([np.sqrt(.3),np.sqrt(.7)*np.exp(.4j)],[1,0,0,0])
        exact=Simulator().run(teleport_circuit(),initial_state=initial,shots=0)
        actual=Simulator(backend='torch',device=device).run(teleport_circuit(),initial_state=initial,shots=0)
        compare_results(exact,actual)
        q=Circuit(2).h(0).measure(0,'m').ry(.7,1).c_if('m',1).reset(0)
        noise=NoiseModel(Noise.amplitude_damping(.17))
        for method in ('auto','trajectory'):
            options=dict(noise=noise,trajectories=257,batch_size=32,shots=0,seed=42)
            reference=Simulator(method=method).run(q,**options)
            actual=Simulator(backend='torch',device=device,method=method).run(q,**options)
            compare_results(reference,actual)
        # Explicit joint-sampling checks on the requested device.
        r=Simulator(backend='torch',device=device,method='trajectory').run(
            Circuit(2).h(0).measure(0,'m').x(1).c_if('m'),trajectories=256,batch_size=32,shots=500,seed=4)
        self.assertTrue(set(r.joint_counts)<=set(['0|00','1|11']))

    def test_torch_cpu_v05(self): self.compare('cpu')

    @unittest.skipUnless(torch is not None and torch.cuda.is_available(),'CUDA GPU unavailable')
    def test_torch_cuda_v05(self): self.compare('cuda:0')


if __name__=='__main__': unittest.main()
