"""Independent numerical checks using Python's standard unittest runner."""
import unittest
import numpy as np
from nexqus import Circuit, Simulator, Noise, NoiseModel, circuit, algorithms


def expanded_operator(matrix, wires, n):
    """Slow basis enumeration used ONLY as an independent small-system reference."""
    d = 2**n
    out = np.zeros((d,d),complex)
    for col in range(d):
        bits = list(format(col,f"0{n}b"))
        subcol = int(''.join(bits[q] for q in wires),2)
        for subrow in range(2**len(wires)):
            result = bits.copy()
            for q,b in zip(wires,format(subrow,f"0{len(wires)}b")): result[q]=b
            out[int(''.join(result),2),col] = matrix[subrow,subcol]
    return out


class SDKTests(unittest.TestCase):
    def setUp(self): self.sim = Simulator()

    def test_bell_state_and_correlations(self):
        r = self.sim.run(algorithms.bell(),shots=0)
        np.testing.assert_allclose(r.statevector,[1/np.sqrt(2),0,0,1/np.sqrt(2)],atol=1e-12)
        self.assertAlmostEqual(r.expectation('XX'),1)
        self.assertAlmostEqual(r.expectation('ZZ'),1)

    def test_ghz(self):
        for n in (1,3,5):
            r=self.sim.run(algorithms.ghz(n),shots=0)
            target=np.zeros(2**n); target[0]=target[-1]=.5
            np.testing.assert_allclose(r.probabilities,target,atol=1e-12)

    def test_qft_all_basis_states(self):
        for n in (1,2,3,4):
            d=2**n
            for k in range(d):
                psi=np.eye(d,dtype=complex)[k]
                r=self.sim.run(algorithms.qft(n),initial_state=psi,shots=0)
                expected=np.exp(2j*np.pi*np.arange(d)*k/d)/np.sqrt(d)
                np.testing.assert_allclose(r.statevector,expected,atol=1e-12)

    def test_qft_inverse_random_state(self):
        rng=np.random.default_rng(8)
        psi=rng.normal(size=16)+1j*rng.normal(size=16); psi/=np.linalg.norm(psi)
        q=algorithms.qft(4).compose(algorithms.qft(4,inverse=True))
        np.testing.assert_allclose(self.sim.run(q,initial_state=psi,shots=0).statevector,psi,atol=1e-12)

    def test_arbitrary_wire_order_against_dense(self):
        rng=np.random.default_rng(12)
        psi=rng.normal(size=8)+1j*rng.normal(size=8); psi/=np.linalg.norm(psi)
        raw=rng.normal(size=(4,4))+1j*rng.normal(size=(4,4)); u,_=np.linalg.qr(raw)
        for wires in ((2,0),(0,2),(1,0)):
            q=Circuit(3).unitary(u,wires)
            expected=expanded_operator(u,wires,3)@psi
            np.testing.assert_allclose(self.sim.run(q,initial_state=psi,shots=0).statevector,expected,atol=1e-12)
            r=Simulator(method='density_matrix').run(q,initial_state=psi,shots=0)
            np.testing.assert_allclose(r.density_matrix,np.outer(expected,expected.conj()),atol=1e-12)

    def test_diagonal_wire_order(self):
        diagonal=np.exp(1j*np.array([.1,.3,.6,.9]))
        psi=np.ones(8)/np.sqrt(8)
        q=Circuit(3).diagonal(diagonal,(2,0))
        u=expanded_operator(np.diag(diagonal),(2,0),3)
        np.testing.assert_allclose(self.sim.run(q,initial_state=psi,shots=0).statevector,u@psi,atol=1e-12)
        np.testing.assert_allclose(Simulator(method='density_matrix').run(q,initial_state=psi,shots=0).density_matrix,
                                   u@np.outer(psi,psi)@u.conj().T,atol=1e-12)

    def test_rotations_and_inverse(self):
        q=Circuit(2).rx(.8,0).ry(.3,1).rz(-.5,0).s(1).t(0).cx(1,0).swap(0,1)
        q.compose(q.inverse())
        np.testing.assert_allclose(self.sim.run(q,shots=0).statevector,[1,0,0,0],atol=1e-12)

    def test_amplitude_damping(self):
        q=Circuit(1).x(0).noise(Noise.amplitude_damping(.3),0)
        r=self.sim.run(q,shots=0)
        np.testing.assert_allclose(r.density_matrix,np.diag([.3,.7]),atol=1e-12)
        self.assertAlmostEqual(r.expectation('Z'),-.4)

    def test_phase_damping(self):
        q=Circuit(1).h(0).noise(Noise.phase_damping(.75),0)
        np.testing.assert_allclose(self.sim.run(q,shots=0).density_matrix,[[.5,.25],[.25,.5]],atol=1e-12)

    def test_depolarizing_convention(self):
        for p in (0,.2,1):
            q=Circuit(1).noise(Noise.depolarizing(p),0)
            np.testing.assert_allclose(self.sim.run(q,shots=0).density_matrix,np.diag([1-p/2,p/2]),atol=1e-12)

    def test_noise_on_entangled_state_dense_reference(self):
        q=algorithms.ghz(3).noise(Noise.amplitude_damping(.25),2).noise(Noise.phase_flip(.2),0)
        psi=np.zeros(8,complex); psi[0]=psi[7]=1/np.sqrt(2)
        expected=np.outer(psi,psi.conj())
        for wire,ch in [(2,Noise.amplitude_damping(.25)),(0,Noise.phase_flip(.2))]:
            ks=[expanded_operator(k,(wire,),3) for k in ch.kraus]
            expected=sum(k@expected@k.conj().T for k in ks)
        rho=self.sim.run(q,shots=0).density_matrix
        np.testing.assert_allclose(rho,expected,atol=1e-12)
        self.assertAlmostEqual(np.trace(rho).real,1)
        self.assertGreaterEqual(np.linalg.eigvalsh(rho).min(),-1e-12)

    def test_noise_rule_selection(self):
        model=NoiseModel().after('X',Noise.bit_flip(1))
        q=Circuit(1).x(0).h(0)
        np.testing.assert_allclose(self.sim.run(q,noise=model,shots=0).density_matrix,[[.5,.5],[.5,.5]],atol=1e-12)

    def test_trajectory_converges_to_exact(self):
        q=algorithms.bell().noise(Noise.amplitude_damping(.35),1).noise(Noise.bit_flip(.2),0)
        expected=self.sim.run(q,shots=0).probabilities
        actual=Simulator(method='trajectory').run(q,shots=0,trajectories=5000,seed=10)
        np.testing.assert_allclose(actual.probabilities,expected,atol=.025)
        self.assertIsNone(actual.statevector)

    def test_noise_endpoints_trajectories(self):
        q=Circuit(1).x(0).noise(Noise.amplitude_damping(1),0)
        np.testing.assert_allclose(Simulator(method='trajectory').run(q,trajectories=10,shots=0).probabilities,[1,0])

    def test_readout_keeps_state(self):
        r=self.sim.run(Circuit(2).x(0),shots=0,readout_error=.1)
        np.testing.assert_allclose(r.probabilities,[.09,.01,.81,.09],atol=1e-12)
        np.testing.assert_allclose(r.statevector,[0,0,1,0])

    def test_grover(self):
        r=self.sim.run(algorithms.grover(2,'10'),shots=0)
        self.assertAlmostEqual(r.probabilities[2],1)
        r=self.sim.run(algorithms.grover(3,['001','110']),shots=0)
        self.assertAlmostEqual(r.probabilities[1]+r.probabilities[6],1)

    def test_bv_and_marginals(self):
        r=self.sim.run(algorithms.bernstein_vazirani('101'),shots=0)
        self.assertAlmostEqual(r.marginal((0,1,2))[5],1)
        self.assertAlmostEqual(r.marginal((2,0,1))[6],1)

    def test_deutsch_jozsa(self):
        self.assertAlmostEqual(self.sim.run(algorithms.deutsch_jozsa([1]*8),shots=0).probabilities[0],1)
        self.assertAlmostEqual(self.sim.run(algorithms.deutsch_jozsa([0,1]*4),shots=0).probabilities[0],0)

    def test_qaoa_against_direct_cost_and_mixer(self):
        gamma,beta=.7,.3
        q=algorithms.qaoa_maxcut(2,[(0,1)],[gamma],[beta])
        cost=np.diag(np.exp(-1j*gamma*np.array([0,1,1,0])))
        x=np.array([[0,1],[1,0]])
        mixer=np.cos(beta)*np.eye(2)-1j*np.sin(beta)*x
        expected=np.kron(mixer,mixer)@cost@np.full(4,.5)
        actual=self.sim.run(q,shots=0).statevector
        # The implemented cost ignores a physically irrelevant global phase.
        np.testing.assert_allclose(np.outer(actual,actual.conj()),np.outer(expected,expected.conj()),atol=1e-12)

    def test_phase_estimation(self):
        for phase in (0,.125,.375,.875):
            u=np.diag([1,np.exp(2j*np.pi*phase)])
            r=self.sim.run(algorithms.phase_estimation(u,3),shots=0)
            self.assertAlmostEqual(r.marginal((0,1,2))[int(phase*8)],1)
        # Use a non-computational eigenstate to check state preparation.
        x=np.array([[0,1],[1,0]],complex)
        r=self.sim.run(algorithms.phase_estimation(x,2,[1/np.sqrt(2),-1/np.sqrt(2)]),shots=0)
        self.assertAlmostEqual(r.marginal((0,1))[2],1)

    def test_function_decorator_and_composition(self):
        @circuit(2)
        def prepare(q,theta): q.ry(theta,0).cx(0,1)
        q=Circuit(3).compose(prepare(np.pi),(2,0))
        self.assertEqual(self.sim.run(q,shots=0).most_likely,'101')

    def test_seeded_counts(self):
        a=self.sim.run(algorithms.bell(),shots=100,seed=4)
        b=self.sim.run(algorithms.bell(),shots=100,seed=4)
        self.assertEqual(a.counts,b.counts)
        self.assertEqual(sum(a.counts.values()),100)
        self.assertTrue(set(a.counts)<=set(['00','11']))

    def test_invalid_inputs(self):
        for n in (0,-1,1.5,True):
            with self.assertRaises(ValueError): Circuit(n)
        for p in (-.1,1.1,float('nan')):
            with self.assertRaises(ValueError): Noise.bit_flip(p)
        with self.assertRaises(ValueError): Circuit(2).cx(0,0)
        with self.assertRaises(ValueError): Circuit(1).x(1)
        with self.assertRaises(ValueError): Circuit(1).unitary(np.ones((2,2)),0)
        with self.assertRaises(ValueError): Noise((np.eye(2)*2,))
        with self.assertRaises(ValueError): algorithms.deutsch_jozsa([0,0,0,1])
        with self.assertRaises(ValueError): self.sim.run(Circuit(1),initial_state=[1,1])
        with self.assertRaises(ValueError): Simulator(method='statevector').run(Circuit(1).noise(Noise.bit_flip(.1),0))
        with self.assertRaises(ValueError): self.sim.run(Circuit(1),shots=-1)
        with self.assertRaises(ValueError): Circuit(1).noise(Noise.bit_flip(.1),0).inverse()

    def test_memory_guard(self):
        with self.assertRaises(MemoryError): Simulator(method='density_matrix',max_memory_mb=1).run(Circuit(12))

    def test_single_precision(self):
        np.testing.assert_allclose(Simulator(precision='single').run(algorithms.ghz(8),shots=100).probabilities,
                                   self.sim.run(algorithms.ghz(8),shots=0).probabilities,atol=1e-6)


try:
    import torch
except ImportError:
    torch=None


@unittest.skipIf(torch is None,"Optional PyTorch not installed")
class TorchTests(unittest.TestCase):
    def compare(self,device):
        q=Circuit(3).h(2).ry(.3,0).cx(2,1).cp(.7,1,0).swap(0,2)
        model=NoiseModel(Noise.depolarizing(.12),Noise.amplitude_damping(.04))
        for method,noise in [('statevector',None),('density_matrix',model),('trajectory',model)]:
            kw=dict(shots=0,seed=9,trajectories=100,noise=noise)
            a=Simulator(method=method).run(q,**kw)
            b=Simulator(backend='torch',device=device,method=method).run(q,**kw)
            np.testing.assert_allclose(a.probabilities,b.probabilities,atol=1e-10)
            if a.statevector is not None: np.testing.assert_allclose(a.statevector,b.statevector,atol=1e-10)
            if a.density_matrix is not None: np.testing.assert_allclose(a.density_matrix,b.density_matrix,atol=1e-10)

    def test_torch_cpu(self): self.compare('cpu')

    @unittest.skipUnless(torch is not None and torch.cuda.is_available(),"CUDA GPU unavailable")
    def test_torch_cuda(self): self.compare('cuda:0')


if __name__ == '__main__': unittest.main()
