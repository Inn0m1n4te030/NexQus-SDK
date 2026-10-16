"""Release regressions and an independent small-system dense oracle.

The oracle scatters local matrix elements into a full computational-basis
operator. It never calls NexQus's backend, kernels, simulator or validator.
"""
import copy
from unittest import mock
import unittest
import numpy as np
from nexqus import (Adam, Circuit, Noise, NoiseModel, Parameter, QuantumFunction,
                    QuantumRegressor, QuantumBinaryClassifier, Simulator, algorithms)
from nexqus.validation import compare_results, trajectory_confidence_bound


def embed(matrix, wires, n):
    full=np.zeros((2**n,2**n),complex)
    for column in range(2**n):
        bits=list(format(column,f'0{n}b'))
        local_column=int(''.join(bits[q] for q in wires),2)
        for local_row in range(2**len(wires)):
            target=bits.copy()
            for q,bit in zip(wires,format(local_row,f'0{len(wires)}b')):
                target[q]=bit
            full[int(''.join(target),2),column]=matrix[local_row,local_column]
    return full


def dense_reference(circuit, psi, noise=None):
    n=circuit.n_qubits; keys=circuit.classical_keys
    branches={tuple(0 for _ in keys):np.outer(psi,psi.conj())}
    def channel(rho,operators,wire):
        result=np.zeros_like(rho)
        for local in operators:
            matrix=embed(local,(wire,),n)
            result+=matrix@rho@matrix.conj().T
        return result
    for op in circuit.operations:
        updated={}
        def add(record,rho):
            updated[record]=updated.get(record,0)+rho
        for record,rho in branches.items():
            if op.condition is not None and record[keys.index(op.condition[0])]!=op.condition[1]:
                add(record,rho)
            elif op.kind=='measure':
                for bit in (0,1):
                    projector=np.zeros((2,2)); projector[bit,bit]=1
                    measured=channel(rho,(projector,),op.wires[0])
                    if np.trace(measured).real>1e-15:
                        changed=list(record); changed[keys.index(op.data)]=bit
                        add(tuple(changed),measured)
            elif op.kind=='reset':
                add(record,channel(rho,(np.diag([1,0]),np.array([[0,1],[0,0]])),op.wires[0]))
            elif op.kind=='noise':
                add(record,channel(rho,op.data.kraus,op.wires[0]))
            else:
                local=np.diag(op.data) if op.kind=='diagonal' else op.data
                matrix=embed(local,op.wires,n)
                changed=matrix@rho@matrix.conj().T
                if noise:
                    for item in noise.channels_for(op.name):
                        for wire in op.wires: changed=channel(changed,item.kraus,wire)
                add(record,changed)
        branches=updated
    density=sum(branches.values())
    classical={''.join(map(str,key)):float(np.trace(rho).real)
               for key,rho in branches.items()} if keys else {}
    return density,classical


def normalized(rng,n):
    psi=rng.normal(size=2**n)+1j*rng.normal(size=2**n)
    return psi/np.linalg.norm(psi)


class DenseReferenceTests(unittest.TestCase):
    def test_random_static_full_density_and_state(self):
        rng=np.random.default_rng(611)
        for index in range(40):
            n=3; q=Circuit(n); psi=normalized(rng,n)
            for _ in range(9):
                count=int(rng.integers(1,4))
                wires=tuple(int(i) for i in rng.choice(n,size=count,replace=False))
                a=rng.normal(size=(2**count,2**count))+1j*rng.normal(size=(2**count,2**count))
                u,_=np.linalg.qr(a)
                q.unitary(u,wires)
                if index%2: q.noise(Noise.amplitude_damping(.17),int(rng.integers(n)))
            model=NoiseModel().after('U',Noise.phase_damping(.13)) if index%3==0 else None
            expected,_=dense_reference(q,psi,model)
            for precision,tol in [('double',2e-12),('single',3e-6)]:
                with self.subTest(index=index,precision=precision):
                    actual=Simulator(precision=precision).run(q,initial_state=psi,noise=model,shots=0)
                    rho=(actual.density_matrix if actual.density_matrix is not None
                         else np.outer(actual.statevector,actual.statevector.conj()))
                    np.testing.assert_allclose(rho,expected,atol=tol,rtol=0)
                    np.testing.assert_allclose(actual.probabilities,expected.diagonal().real,atol=tol,rtol=0)

    def test_random_dynamic_full_density_and_records(self):
        rng=np.random.default_rng(612)
        for index in range(24):
            q=Circuit(3).h(0).cx(0,2).measure(0,'a')
            psi=normalized(rng,3)
            for _ in range(10):
                wire=int(rng.integers(3)); choice=int(rng.integers(5))
                if choice==0: q.measure(wire,'b' if rng.random()<.5 else 'a')
                elif choice==1: q.reset(wire)
                elif choice==2: q.noise(Noise.amplitude_damping(.2),wire)
                elif choice==3: q.ry(float(rng.normal()),wire)
                else: q.cx(*map(int,rng.choice(3,size=2,replace=False)))
                if choice!=0 and rng.random()<.5: q.c_if('a',int(rng.integers(2)))
            model=NoiseModel(Noise.depolarizing(.07))
            expected,records=dense_reference(q,psi,model)
            actual=Simulator().run(q,noise=model,initial_state=psi,shots=0)
            with self.subTest(index=index):
                np.testing.assert_allclose(actual.density_matrix,expected,atol=2e-12)
                np.testing.assert_allclose(actual.probabilities,expected.diagonal().real,atol=2e-12)
                self.assertEqual(set(records),set(actual.classical_probabilities))
                for key in records: self.assertAlmostEqual(records[key],actual.classical_probabilities[key],places=12)

    def test_sampled_custom_channel_against_dense(self):
        q=Circuit(2).h(1).ry(.9,0).cx(1,0).cp(.7,0,1)
        rotation=np.array([[1,1j],[1j,1]])/np.sqrt(2)
        custom=Noise(tuple(rotation@k@rotation.conj().T for k in Noise.amplitude_damping(.31).kraus))
        q.noise(custom,1).h(0)
        psi=np.array([1,0,0,0],complex)
        expected,_=dense_reference(q,psi)
        actual=Simulator(method='trajectory').run(q,shots=0,trajectories=6000,batch_size=127,seed=17)
        limit=trajectory_confidence_bound(6000,4,alpha=.001)
        self.assertLess(np.max(abs(actual.probabilities-expected.diagonal().real)),limit)

    def test_sampled_dynamic_against_dense(self):
        q=Circuit(2).h(0).cx(0,1).measure(0,'a').reset(1).c_if('a',1)
        q.h(1).measure(1,'b').x(0).c_if('b',0).noise(Noise.phase_damping(.2),0)
        model=NoiseModel(Noise.amplitude_damping(.13))
        expected,records=dense_reference(q,np.array([1,0,0,0],complex),model)
        actual=Simulator(method='trajectory').run(q,noise=model,shots=800,trajectories=6000,batch_size=127,seed=18)
        limit=trajectory_confidence_bound(6000,8,alpha=.001)
        self.assertLess(np.max(abs(actual.probabilities-expected.diagonal().real)),limit)
        for key in records:
            self.assertLess(abs(records[key]-actual.classical_probabilities.get(key,0)),limit)
        self.assertEqual(sum(actual.joint_counts.values()),800)


class InputRegressionTests(unittest.TestCase):
    def test_single_precision_noise_states_remain_complex64(self):
        for channel in (Noise.amplitude_damping(.2),Noise.phase_damping(.3),Noise.depolarizing(.2)):
            for fast in (True,False):
                sim=Simulator(method='trajectory',precision='single')
                original=sim.backend.stack
                original_apply=sim.backend.apply_batch
                observed=[]
                def checked(values,axis=0):
                    result=original(values,axis)
                    if np.iscomplexobj(result): observed.append(result.dtype)
                    return result
                def checked_apply(*args,**kwargs):
                    result=original_apply(*args,**kwargs)
                    if np.iscomplexobj(result): observed.append(result.dtype)
                    return result
                sim.backend.stack=checked
                sim.backend.apply_batch=checked_apply
                q=Circuit(2).ry(.7,0).cx(0,1).noise(channel,0)
                result=sim.run(q,trajectories=7,batch_size=4,noise_fastpath=fast,seed=1,shots=0)
                self.assertTrue(observed)
                self.assertTrue(all(dtype==np.dtype('complex64') for dtype in observed))
                self.assertEqual(result.metadata['state_storage_bytes'],4*8)

    def test_nonunitary_diagonals_rejected(self):
        for value in (1+1e-6,1-1e-6,np.inf,np.nan):
            with self.subTest(value=value),self.assertRaises(ValueError):
                Circuit(1).diagonal([1,value],0)

    def test_bad_initial_norm_all_modes(self):
        for q in (Circuit(1),Circuit(1).reset(0)):
            for method in ('auto','density_matrix','trajectory'):
                with self.subTest(dynamic=q.dynamic,method=method),self.assertRaises(ValueError):
                    Simulator(method=method).run(q,initial_state=[1+1e-6,0],shots=0)

    def test_initial_roundoff_removed_and_input_unchanged(self):
        psi=np.array([1+1e-12,0],complex); saved=psi.copy()
        result=Simulator().run(Circuit(1),initial_state=psi,shots=0)
        self.assertAlmostEqual(np.linalg.norm(result.statevector),1,places=15)
        np.testing.assert_array_equal(psi,saved)

    def test_qpe_rejects_bad_target_norm(self):
        with self.assertRaises(ValueError): algorithms.phase_estimation(np.eye(2),target_state=[1+1e-6,0])

    def test_scalar_marginal_and_invalid_wires(self):
        result=Simulator().run(Circuit(2).x(0),shots=0)
        np.testing.assert_array_equal(result.marginal(0),[0,1])
        for wires in (True,None,1.5,(0,0)):
            with self.assertRaises(ValueError): result.marginal(wires)

    def test_dynamic_history_memory_is_not_linear_in_trajectories(self):
        # Stop at the first evolution operation. The old implementation tried
        # to allocate an 8 TB shot-schedule array before reaching this point.
        class ReachedEvolution(Exception): pass
        original=np.bincount
        def guarded(*args,**kwargs):
            if kwargs.get('minlength',0)>1000:
                self.fail('Host allocation scales with total trajectory count')
            return original(*args,**kwargs)
        sim=Simulator(method='trajectory')
        q=Circuit(1).h(0).measure(0,'m')
        with mock.patch('numpy.bincount',side_effect=guarded), \
             mock.patch.object(sim.backend,'apply_batch',side_effect=ReachedEvolution):
            with self.assertRaises(ReachedEvolution):
                sim.run(q,trajectories=10**12,batch_size=1,shots=0,seed=1)

    def test_mutated_dynamic_quantum_function_rejected(self):
        theta=Parameter('theta'); template=Circuit(1).ry(theta,0)
        function=QuantumFunction(template,'Z')
        template.measure(0,'m')
        with self.assertRaises(ValueError): function({'theta':.3})
        with self.assertRaises(ValueError): function.gradient({'theta':.3})

    def test_complex_learning_data_rejected(self):
        x=Parameter('x'); w=Parameter('w'); f=QuantumFunction(Circuit(1).ry(w*x,0),'Z')
        for model in (QuantumRegressor(f,'x',{'w':.1}),QuantumBinaryClassifier(f,'x',{'w':.1})):
            with self.assertRaises(ValueError): model.predict(np.array([1+2j]))
            with self.assertRaises(ValueError): model.fit([1],np.array([1+2j]),epochs=1)

    def test_callback_cannot_mutate_history_parameters(self):
        x=Parameter('x'); w=Parameter('w')
        model=QuantumRegressor(QuantumFunction(Circuit(1).ry(w*x,0),'Z'),'x',{'w':.2})
        def callback(record): record['parameters']['w']=999
        model.fit([1],[0],epochs=1,callback=callback)
        self.assertEqual(model.history[0]['parameters'],{'w':.2})

    def test_adam_failed_step_is_atomic(self):
        adam=Adam(); adam.step({'a':1,'b':1},{'a':.1,'b':.2})
        before=copy.deepcopy((adam.t,adam.m,adam.v))
        for gradient in (1e155,1e308):
            with self.assertRaises(ValueError): adam.step({'a':1,'b':1},{'a':.1,'b':gradient})
            self.assertEqual(before,(adam.t,adam.m,adam.v))


class ValidationRegressionTests(unittest.TestCase):
    def test_nonfinite_retained_state_rejected(self):
        for method,field in [('statevector','statevector'),('density_matrix','density_matrix')]:
            reference=Simulator(method=method).run(algorithms.bell(),shots=0)
            for value in (np.nan,np.inf):
                candidate=copy.deepcopy(reference); getattr(candidate,field).flat[0]=value
                with self.assertRaises(AssertionError): compare_results(reference,candidate)

    def test_missing_or_wrong_state_shape_rejected(self):
        reference=Simulator().run(algorithms.bell(),shots=0)
        for state in (None,np.ones((4,1))/2,np.ones(4)):
            candidate=copy.deepcopy(reference); candidate.statevector=state
            with self.assertRaises(AssertionError): compare_results(reference,candidate)

    def test_nonfinite_classical_probability_rejected(self):
        reference=Simulator().run(Circuit(1).h(0).measure(0,'m'),shots=0)
        for value in (np.nan,np.inf,-.1):
            candidate=copy.deepcopy(reference); candidate.classical_probabilities['0']=value
            with self.assertRaises(AssertionError): compare_results(reference,candidate)

    def test_unseeded_trajectory_parity_rejected(self):
        result=Simulator(method='trajectory').run(Circuit(1).h(0),shots=0,trajectories=2)
        with self.assertRaises(ValueError): compare_results(result,result)

    def test_bad_confidence_arguments_rejected(self):
        for n,k,alpha in [(1.5,2,.01),(True,2,.01),(3,1.5,.01),(3,2,np.nan)]:
            with self.assertRaises(ValueError): trajectory_confidence_bound(n,k,alpha)


try:
    import torch
except ImportError:
    torch=None


@unittest.skipIf(torch is None,'Optional PyTorch not installed')
class ReleaseTorchTests(unittest.TestCase):
    def check_device(self,device):
        for precision in ('double','single'):
            for dynamic in (False,True):
                q=Circuit(2).h(0).cx(0,1).ry(.3,1)
                if dynamic: q.measure(0,'m').reset(1).c_if('m',1)
                for method in ('auto','trajectory'):
                    options=dict(seed=41,shots=31,trajectories=129,batch_size=17,
                                 noise=NoiseModel(Noise.amplitude_damping(.2)))
                    reference=Simulator(method=method,precision=precision).run(q,**options)
                    actual=Simulator(backend='torch',device=device,method=method,precision=precision).compile(q,options.pop('noise')).run(**options)
                    compare_results(reference,actual,precision=precision)
                    self.assertEqual(actual.metadata['device'],device)
                    self.assertEqual(sum(actual.counts.values()),31)

    def test_release_torch_cpu(self): self.check_device('cpu')

    @unittest.skipUnless(torch is not None and torch.cuda.is_available(),'CUDA GPU unavailable')
    def test_release_cuda(self): self.check_device('cuda:0')


if __name__=='__main__': unittest.main()
