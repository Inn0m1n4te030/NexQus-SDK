"""NumPy/Torch oracle comparisons; CUDA tests require actual hardware."""
import os
import sys
import types
import unittest
from unittest.mock import patch
import numpy as np
from nexqus import Circuit,Parameter,Simulator,QuantumFunction,QuantumDistribution,QGAN,QuantumRegressor,Adam,Noise,NoiseModel,algorithms,distribution_metrics
from nexqus.chemistry import h2_reference_problem,h2_pair_ansatz
from nexqus.backend import Backend
try:
    import torch
except ImportError: torch=None


@unittest.skipIf(torch is None,'Optional PyTorch not installed')
class TorchWorkflows(unittest.TestCase):
    device='cpu'

    def test_native_results_noise_dynamic_and_reductions(self):
        sim=Simulator.for_device(self.device,backend='torch')
        q=Circuit(3).h(2).ry(.47,0).cx(2,0).rx(.2,1)
        for circuit,noise in ((q,None),(q,NoiseModel(Noise.amplitude_damping(.13))),
                              (Circuit(2).h(0).cx(0,1).measure(0,'m').reset(1),None)):
            host=Simulator(backend='numpy').run(circuit,shots=0,noise=noise)
            actual=sim.run(circuit,shots=0,noise=noise,native_result=True)
            self.assertEqual(actual.probabilities.device.type,torch.device(self.device).type)
            np.testing.assert_allclose(actual.to_numpy().probabilities,host.probabilities,atol=1e-12)
            for word in ('X'+'Y'*(circuit.n_qubits-1),'Z'*circuit.n_qubits,'I'*circuit.n_qubits):
                self.assertAlmostEqual(actual.expectation(word),host.expectation(word),places=11)
            for wires in ([0],[circuit.n_qubits-1,0]):
                np.testing.assert_allclose(actual.reduced_density(wires).detach().cpu(),host.reduced_density(wires),atol=1e-12)
                np.testing.assert_allclose(actual.marginal(wires).detach().cpu(),host.marginal(wires),atol=1e-12)
        compiled=sim.compile(q)
        self.assertTrue(torch.is_tensor(compiled.run(shots=0,native_result=True).statevector))

    def test_gradients_and_vqe_without_host_state_export(self):
        angle=Parameter('theta'); q=h2_pair_ansatz(angle); problem=h2_reference_problem()
        f=problem.hamiltonian.function(q,backend='torch',device=self.device)
        reference=problem.hamiltonian.function(q,backend='numpy',device='cpu')
        with patch.object(Backend,'numpy',side_effect=AssertionError('Unexpected full-state host export')):
            v,g=f.value_and_grad({'theta':.4})
            pure=QuantumFunction(Circuit(1).ry(angle,0),'Z',backend='torch',device=self.device)
            self.assertAlmostEqual(pure({'theta':.4}),np.cos(.4))
        rv,rg=reference.value_and_grad({'theta':.4})
        self.assertAlmostEqual(v,rv,places=11); self.assertAlmostEqual(g['theta'],rg['theta'],places=11)
        self.assertAlmostEqual(problem.exact_energy(backend='torch',device=self.device),problem.exact_energy(device='cpu'),places=11)
        params={'theta':0.}; optimizer=Adam(.04)
        for _ in range(180): params=optimizer.step(params,f.gradient(params))
        self.assertLess(abs(f(params)-problem.exact_energy(device='cpu')),2e-7)

    def test_distribution_qgan_learning_and_shor(self):
        t=Parameter('t'); q=Circuit(2).ry(t,0).cx(0,1)
        native=QuantumDistribution(q,backend='torch',device=self.device,native=True)
        cpu=QuantumDistribution(q,backend='numpy',device='cpu')
        p,g=native.value_and_grad({'t':.9})
        np.testing.assert_allclose(p.detach().cpu(),cpu({'t':.9}),atol=1e-12)
        self.assertAlmostEqual(distribution_metrics(cpu({'t':.9}),p)['total_variation'],0,places=11)
        np.testing.assert_allclose(g['t'].detach().cpu(),cpu.gradient({'t':.9})['t'],atol=1e-12)
        zero=QuantumDistribution(Circuit(1).ry(t*t,0),backend='torch',device=self.device,native=True)
        torch.testing.assert_close(zero.gradient({'t':0})['t'],torch.zeros(2,dtype=torch.float64,device=self.device))
        generator=QuantumDistribution(Circuit(1).ry(t,0),backend='torch',device=self.device)
        model=QGAN(generator,{'t':1.3}); model.fit([.25,.75],epochs=40)
        oracle=QGAN(QuantumDistribution(Circuit(1).ry(t,0),backend='numpy',device='cpu'),{'t':1.3}).fit([.25,.75],epochs=40)
        np.testing.assert_allclose(model.probabilities(),oracle.probabilities(),atol=1e-10)
        x,w=Parameter('x'),Parameter('w')
        f=QuantumFunction(Circuit(1).ry(x*w,0),'Z',backend='torch',device=self.device)
        regressor=QuantumRegressor(f,['x'],{'w':.2})
        samples=np.linspace(-1,1,9); target=np.cos(samples*.8)
        before=np.mean((regressor.predict(samples)-target)**2)
        regressor.fit(samples,target,epochs=20)
        self.assertLess(np.mean((regressor.predict(samples)-target)**2),before)
        result=algorithms.shor(15,base=2,seed=42,simulator=Simulator.for_device(self.device,backend='torch'))
        self.assertEqual(result.factors,(3,5))

    def test_batched_symbolic_autograd_matches_scalar_oracle(self):
        from nexqus.torch_circuit import TorchCircuit
        t,u=Parameter('t'),Parameter('u')
        q=Circuit(3).h(2).rx(t*.7,0).ry(t*t+u,1).cx(2,0).cp(t/(u+2),2,1).phase(u,0).rz(u,2).h(1)
        q.permutation([1,2,3,0],[2,0])
        layer=TorchCircuit(q,device=self.device)
        bindings={'t':torch.tensor([.4,.7],dtype=torch.double,device=self.device,requires_grad=True),
                  'u':torch.tensor(-.2,dtype=torch.double,device=self.device,requires_grad=True)}
        p=layer(bindings)
        reference=QuantumDistribution(q,backend='numpy',device='cpu')
        for i,tvalue in enumerate((.4,.7)):
            expected,jac=reference.value_and_grad({'t':tvalue,'u':-.2})
            np.testing.assert_allclose(p[i].detach().cpu(),expected,atol=1e-12)
        coefficients=torch.arange(8,dtype=torch.double,device=self.device)
        gt,gu=torch.autograd.grad((p*coefficients).sum(),tuple(bindings.values()))
        expected_t=[]; expected_u=0
        for tvalue in (.4,.7):
            _,jac=reference.value_and_grad({'t':tvalue,'u':-.2})
            expected_t.append(float(jac['t']@np.arange(8))); expected_u+=float(jac['u']@np.arange(8))
        np.testing.assert_allclose(gt.cpu(),expected_t,atol=1e-11); self.assertAlmostEqual(float(gu),expected_u,places=11)
        with self.assertRaises(MemoryError): TorchCircuit(q,max_memory_mb=.0001,device=self.device)(bindings)
        with self.assertRaises(ValueError): layer({'t':bindings['t'],'u':torch.tensor(1j,device=self.device)})

    def test_explanation_execution_device_and_export(self):
        data=Circuit(2).h(0).cx(0,1).explain(backend='torch',device=self.device).to_dict()
        self.assertEqual(data['execution']['device'],self.device)
        np.testing.assert_allclose(data['frames'][-1]['probabilities'],[.5,0,0,.5],atol=1e-12)


@unittest.skipUnless(torch is not None and torch.cuda.is_available(),'CUDA GPU unavailable')
class CUDAWorkflows(TorchWorkflows):
    device=os.environ.get('NEXQUS_TEST_CUDA_DEVICE','cuda:0')


class DeviceSelectionTests(unittest.TestCase):
    def test_scf_gpu_dispatch_and_explicit_host_integral_stage(self):
        # Dispatch regression only; this mock is not GPU4PySCF hardware evidence.
        from nexqus.chemistry import molecular_hamiltonian
        events=[]
        class Mol:
            nelectron=2
            def nao_nr(self): return 1
            def energy_nuc(self): return .5
        class Field:
            mo_coeff=np.eye(1); converged=True
            def kernel(self): events.append('kernel'); return -1.
            def get_hcore(self): return np.array([[-1.]])
            def to_gpu(self): events.append('to_gpu'); return self
            def to_cpu(self): events.append('to_cpu'); return self
        class Device:
            def __init__(self,index): self.index=index
            def __enter__(self): events.append(('device',self.index))
            def __exit__(self,*args): pass
        pyscf=types.ModuleType('pyscf')
        pyscf.gto=types.SimpleNamespace(M=lambda **kw:Mol())
        pyscf.scf=types.SimpleNamespace(RHF=lambda mol:Field())
        pyscf.ao2mo=types.SimpleNamespace(kernel=lambda *args,**kw:np.array([.6]))
        cupy=types.ModuleType('cupy')
        cupy.cuda=types.SimpleNamespace(Device=Device,runtime=types.SimpleNamespace(getDeviceCount=lambda:2))
        with patch.dict(sys.modules,{'pyscf':pyscf,'cupy':cupy,'gpu4pyscf':types.ModuleType('gpu4pyscf')}):
            problem=molecular_hamiltonian(['H','H'],[[0,0,0],[0,0,.7]],scf_device='cuda:1')
        self.assertEqual(events,[('device',1),'to_gpu','kernel','to_cpu'])
        self.assertEqual(problem.scf_device,'cuda:1')

    def test_environment_and_explicit_cpu(self):
        with patch.dict(os.environ,{'NEXQUS_DEVICE':'invalid'}):
            with self.assertRaises(ValueError): Simulator()
            self.assertEqual(Simulator(device='cpu').backend.name,'numpy')
        with self.assertRaises(ValueError): QuantumFunction(Circuit(1),'Z',Simulator(device='cpu'),device='cpu')
        if torch is not None and not torch.cuda.is_available():
            with self.assertRaisesRegex(RuntimeError,'no silent CPU fallback'): Simulator.for_device('cuda')


if __name__=='__main__': unittest.main()
