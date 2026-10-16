"""Statevector, density-matrix and Monte Carlo trajectory simulation."""
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from math import sqrt
import copy
import hashlib
import time
import numpy as np
from .backend import Backend
from .circuit import Circuit, positive_integer, I, X, Y, Z
from .noise import NoiseModel
from .random import DeviceRandom


def _initial_vector(initial_state, size):
    """Validate once for static and dynamic modes, then remove roundoff in norm."""
    if initial_state is None:
        psi = np.zeros(size, complex)
        psi[0] = 1
        return psi
    psi = np.asarray(initial_state, dtype=complex)
    if psi.shape != (size,) or not np.isfinite(psi).all():
        raise ValueError("initial_state must be a finite normalized statevector")
    with np.errstate(over='ignore', invalid='ignore'):
        norm = np.vdot(psi, psi).real
    if not np.isfinite(norm) or not np.isclose(norm, 1, atol=1e-10, rtol=0):
        raise ValueError("initial_state squared norm must equal 1 within 1e-10")
    return psi / np.sqrt(norm)


class CompiledCircuit:
    """A circuit/noise snapshot with device tensors prepared for one Simulator.

    This is execution preparation, not native CUDA code generation. Source
    circuit edits do not affect the compiled snapshot. Call .run() repeatedly.
    """
    def __init__(self, owner, circuit, noise, arrays, seconds):
        self._owner=owner
        self._circuit=circuit
        self._noise=noise
        self._arrays=arrays
        self.compile_seconds=seconds

    @property
    def n_qubits(self): return self._circuit.n_qubits

    def run(self, **kwargs): return self._owner.run(self, **kwargs)


@dataclass
class Result:
    probabilities: np.ndarray
    counts: dict
    metadata: dict
    statevector: object = None
    density_matrix: object = None
    classical_keys: tuple = ()
    classical_probabilities: dict = field(default_factory=dict)
    classical_counts: dict = field(default_factory=dict)
    joint_counts: dict = field(default_factory=dict)

    def reduced_density(self,wires):
        """Reduced quantum state, in the supplied wire order; requires retained state."""
        n=self.metadata['n_qubits']; wires=Circuit(n)._wires(wires)
        rest=tuple(q for q in range(n) if q not in wires)
        d=2**len(wires); t=2**len(rest)
        if self.statevector is not None:
            matrix=self.statevector.reshape((2,)*n).transpose(wires+rest).reshape(d,t)
            return matrix@matrix.conj().T
        if self.density_matrix is not None:
            axes=wires+rest+tuple(q+n for q in wires+rest)
            tensor=self.density_matrix.reshape((2,)*(2*n)).transpose(axes).reshape(d,t,d,t)
            return np.trace(tensor,axis1=1,axis2=3)
        raise ValueError('Reduced density requires a retained statevector or density matrix')

    @property
    def most_likely(self):
        return format(int(self.probabilities.argmax()), f"0{self.metadata['n_qubits']}b")

    def marginal(self, wires):
        n = self.metadata["n_qubits"]
        wires = Circuit(n)._wires(wires)
        axes = wires + tuple(q for q in range(n) if q not in wires)
        return self.probabilities.reshape((2,)*n).transpose(axes).reshape(2**len(wires), -1).sum(axis=1)

    def expectation(self, pauli):
        """Pauli string in q0..qn order, e.g. 'ZI' or 'XX'.

        Requires retain_state=True. Trajectory runs do not retain a mixed state.
        """
        n = self.metadata["n_qubits"]
        if not isinstance(pauli, str) or len(pauli) != n or any(p not in "IXYZ" for p in pauli):
            raise ValueError("Provide one I/X/Y/Z character per qubit")
        if self.statevector is None and self.density_matrix is None:
            raise ValueError("Expectation requires a retained statevector or density matrix")
        backend = Backend()
        matrices = {"I": I, "X": X, "Y": Y, "Z": Z}
        density = self.density_matrix is not None
        original = self.density_matrix if density else self.statevector
        work = original.reshape((2,)*(2*n if density else n))
        for q, p in enumerate(pauli):
            if p != "I": work = backend.apply(work, matrices[p], (q,))
        if density: return float(np.trace(work.reshape(2**n, 2**n)).real)
        return float(np.vdot(original, work.reshape(-1)).real)


class Simulator:
    """device='cpu' by default; CUDA is opt-in via backend='torch'.

    method='auto': statevector without noise, density_matrix with noise.
    method='trajectory': finite-sample noisy ensemble, configurable batch size.
    max_memory_mb limits estimated core working storage, not process RAM.
    """
    def __init__(self, backend="numpy", device="cpu", method="auto", precision="double",
                 max_memory_mb=1024, gate_cache_mb=64, local_gates=True):
        if method not in ("auto", "statevector", "density_matrix", "trajectory"):
            raise ValueError("Unknown simulation method")
        if not np.isfinite(max_memory_mb) or max_memory_mb <= 0:
            raise ValueError("max_memory_mb must be positive")
        self.backend = Backend(backend, device, precision, local_gates=local_gates)
        self.method = method
        self.precision = precision
        self.max_memory_mb = float(max_memory_mb)
        if not np.isfinite(gate_cache_mb) or gate_cache_mb < 0:
            raise ValueError('gate_cache_mb must be nonnegative')
        self._cache_limit=int(gate_cache_mb*1024**2)
        self._gate_cache=OrderedDict()
        self._cache_bytes=0
        self._cache_hits=self._cache_misses=0

    def _prepare_array(self, data, permutation=False):
        contiguous=np.ascontiguousarray(data)
        key=(permutation,contiguous.shape,contiguous.dtype.str,
             hashlib.blake2b(contiguous.tobytes(),digest_size=16).digest())
        if key in self._gate_cache:
            self._cache_hits+=1
            self._gate_cache.move_to_end(key)
            return self._gate_cache[key][0]
        self._cache_misses+=1
        native=(self.backend.index_array(np.argsort(data)) if permutation
                else self.backend.array(data))
        size=int(contiguous.size)*(8 if permutation else self.backend.itemsize)
        if size <= self._cache_limit:
            while self._cache_bytes+size > self._cache_limit:
                _,(_,removed)=self._gate_cache.popitem(last=False)
                self._cache_bytes-=removed
            self._gate_cache[key]=(native,size)
            self._cache_bytes+=size
        return native

    def clear_cache(self):
        """Drop simulator cache references; live compiled objects still own tensors."""
        self._gate_cache.clear()
        self._cache_bytes=0
        self._cache_hits=self._cache_misses=0

    def cache_info(self):
        return dict(entries=len(self._gate_cache),bytes=self._cache_bytes,
                    limit_bytes=self._cache_limit,hits=self._cache_hits,misses=self._cache_misses)

    def compile(self, circuit, noise=None):
        """Snapshot a concrete circuit and prepare/deduplicate its device tensors."""
        started=time.perf_counter()
        if not isinstance(circuit,Circuit): raise TypeError('Expected Circuit')
        if circuit.parameters: raise ValueError('Bind circuit parameters before compilation')
        if noise is not None and not isinstance(noise,NoiseModel): raise TypeError('Expected NoiseModel')
        snapshot=copy.deepcopy(circuit)
        frozen_noise=copy.deepcopy(noise)
        arrays={}
        def prepare(data, permutation=False):
            if id(data) not in arrays: arrays[id(data)]=self._prepare_array(data, permutation)
        for op in snapshot.operations:
            if op.kind=='noise':
                for k in op.data.kraus: prepare(k)
            elif op.kind in ('unitary','diagonal','permutation'): prepare(op.data, op.kind=='permutation')
        if frozen_noise:
            for _,ch in frozen_noise.rules:
                for k in ch.kraus: prepare(k)
        self.backend.sync()
        return CompiledCircuit(self,snapshot,frozen_noise,arrays,time.perf_counter()-started)

    def run_sweep(self, template, bindings, seed=None, **run_options):
        """Run bindings in input order, with independent reproducible child seeds.

        Results are accumulated in a list. For large sweeps, use iter_sweep().
        State retention defaults to False to avoid storing every full state.
        """
        return list(self.iter_sweep(template,bindings,seed=seed,**run_options))

    def iter_sweep(self, template, bindings, seed=None, **run_options):
        if not isinstance(template,Circuit): raise TypeError('Expected parameterized Circuit')
        sequence=np.random.SeedSequence(seed)
        options=dict(run_options)
        options.setdefault('retain_state',False)
        for index,values in enumerate(bindings):
            bound=template.bind(values)
            child_seed=int(sequence.spawn(1)[0].generate_state(1)[0])
            result=self.run(bound,seed=child_seed,**options)
            from .parameters import bindings_dict
            result.metadata['sweep_index']=index
            result.metadata['bindings']=bindings_dict(values)
            yield result

    def run(self, circuit, shots=1024, seed=None, noise=None, trajectories=256,
            initial_state=None, retain_state=True, readout_error=0.0, batch_size=1,
            random_engine='portable', noise_fastpath=True, max_branches=1024):
        prepared=isinstance(circuit,CompiledCircuit)
        prepared_arrays={}
        compile_seconds=0.0
        if prepared:
            if circuit._owner is not self: raise ValueError('CompiledCircuit belongs to another Simulator')
            if noise is not None: raise ValueError('Noise is frozen at compile time; recompile to change it')
            noise=circuit._noise
            prepared_arrays=circuit._arrays
            compile_seconds=circuit.compile_seconds
            circuit=circuit._circuit
        if not isinstance(circuit, Circuit): raise TypeError("Expected a Circuit")
        if circuit.parameters: raise ValueError('Unbound parameters; use circuit.bind() or run_sweep()')
        if circuit.dynamic:
            from .dynamic import run_dynamic
            return run_dynamic(self,circuit,shots=shots,seed=seed,noise=noise,
                               trajectories=trajectories,initial_state=initial_state,
                               retain_state=retain_state,readout_error=readout_error,
                               batch_size=batch_size,random_engine=random_engine,
                               max_branches=max_branches,prepared_arrays=prepared_arrays,
                               compiled=prepared,compile_seconds=compile_seconds)
        if isinstance(shots, bool) or not isinstance(shots, (int, np.integer)) or shots < 0:
            raise ValueError("shots must be a nonnegative integer")
        if noise is not None and not isinstance(noise, NoiseModel):
            raise TypeError("noise must be a NoiseModel")
        if not np.isfinite(readout_error) or not 0 <= readout_error <= 1:
            raise ValueError("readout_error must be in [0,1]")
        noisy = any(op.kind == "noise" for op in circuit.operations) or bool(noise and noise.rules)
        method = ("density_matrix" if noisy else "statevector") if self.method == "auto" else self.method
        if noisy and method == "statevector":
            raise ValueError("Noise requires density_matrix, trajectory, or auto")
        n = circuit.n_qubits
        size = 2**n
        batch_size=positive_integer(batch_size,'batch_size')
        if method=='trajectory':
            trajectories=positive_integer(trajectories,'trajectories')
        effective_batch=min(batch_size,trajectories) if method=='trajectory' else 1
        base_bytes = self.backend.itemsize * (size**2 if method == "density_matrix" else size)
        # Working tensor, result, transpose/matmul temporaries; not a hard allocator cap.
        working_bytes = base_bytes * 6 * effective_batch
        if working_bytes > self.max_memory_mb * 1024**2:
            raise MemoryError(f"Estimated working tensors need {working_bytes/1024**2:.1f} MiB; "
                              "increase max_memory_mb or use fewer qubits/trajectory mode")
        psi = _initial_vector(initial_state, size)
        rng = np.random.default_rng(seed)
        b = self.backend
        device_rng=DeviceRandom(b,seed,random_engine)
        channel_cache={}
        damping_cache={}
        def prepare_channel(ch):
            if id(ch) not in channel_cache:
                weights=ch.pauli_weights() if noise_fastpath else None
                channel_cache[id(ch)]=None if weights is None else b.real_array(weights)
                damping_cache[id(ch)]=ch.damping_spec() if noise_fastpath and weights is None else None
        for op in circuit.operations:
            if op.kind=='noise': prepare_channel(op.data)
        if noise:
            for _,ch in noise.rules: prepare_channel(ch)
        cache = dict(prepared_arrays)
        cx_matrix=np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]],complex)
        cx_data={id(op.data) for op in circuit.operations if b.local_gates and
                 op.kind=='unitary' and len(op.wires)==2 and np.array_equal(op.data,cx_matrix)}
        def array(data, permutation=False):
            key = id(data)
            if key not in cache: cache[key] = self._prepare_array(data, permutation)
            return cache[key]

        def transform(state, data, wires, density=False, diagonal=False, permutation=False):
            if id(data) in cx_data:
                state=b.apply_cx(state,wires[0],wires[1])
                if density: state=b.apply_cx(state,n+wires[0],n+wires[1])
                return state
            native = array(data, permutation)
            state = b.apply(state, native, wires, diagonal, permutation)
            if density:
                state = b.apply(state, native if permutation else native.conj(), tuple(n+q for q in wires), diagonal, permutation)
            return state

        def channel(state, ch, q, density):
            if not density: raise RuntimeError('Quantum noise needs density or trajectory mode')
            total = b.zeros(state.shape)
            for k in ch.kraus: total += transform(state, k, (q,), True)
            return total

        def evolve(density):
            start=initial_vector[:,None]*initial_vector.conj()[None,:] if density else initial_vector
            state=start.reshape((2,)*(2*n if density else n))
            for op in circuit.operations:
                if op.kind == "noise":
                    state = channel(state, op.data, op.wires[0], density)
                    continue
                state = transform(state, op.data, op.wires, density, op.kind == "diagonal", op.kind == "permutation")
                if noise:
                    for ch in noise.channels_for(op.name):
                        for q in op.wires: state = channel(state, ch, q, density)
            return state

        def batch_channel(state,ch,q):
            """Branch weights, inverse-CDF sampling and selection stay on device."""
            count=state.shape[0]
            u=device_rng.uniform(count)
            pauli=channel_cache[id(ch)]
            if pauli is not None:
                cumulative=b.cumulative(pauli,0)
                choices=(u[:,None]>=cumulative[None,:-1]).sum(axis=1)
                # Skip identity application and all Kraus norm calculations.
                x=state.reshape(count,2**q,2,2**(n-q-1))
                flip=((choices==1)|(choices==2)).reshape(count,1,1)
                a=b.where(flip,x[:,:,1,:],x[:,:,0,:])
                d=b.where(flip,x[:,:,0,:],x[:,:,1,:])
                y=(choices==2).reshape(count,1,1)
                z=(choices==3).reshape(count,1,1)
                a=b.where(y,-1j*a,a)
                d=b.where(y,1j*d,b.where(z,-d,d))
                return b.stack([a,d],axis=2).reshape(state.shape)
            damping=damping_cache[id(ch)]
            if damping is not None:
                kind,gamma=damping
                x=state.reshape(count,2**q,2,2**(n-q-1))
                a=x[:,:,0,:]; d=x[:,:,1,:]
                pop0=(a.conj()*a).real.sum(axis=(1,2))
                pop1=(d.conj()*d).real.sum(axis=(1,2))
                nojump=pop0+(1-gamma)*pop1
                jump=(u>=nojump/b.clamp_min(pop0+pop1,1e-30)).reshape(count,1,1)
                scale0=1/b.clamp_min(nojump,1e-30)**.5
                scale1=1/b.clamp_min(pop1,1e-30)**.5
                first=a*scale0[:,None,None]
                # A Python scalar preserves complex64. A NumPy float64 scalar
                # promotes this whole batch to complex128 under NumPy 2.x.
                second=d*sqrt(1-gamma)*scale0[:,None,None]
                if kind=='amplitude':
                    first=b.where(jump,d*scale1[:,None,None],first)
                    second=b.where(jump,0,second)
                else:
                    first=b.where(jump,0,first)
                    second=b.where(jump,d*scale1[:,None,None],second)
                return b.stack([first,second],axis=2).reshape(state.shape)
            weights=[]
            for k in ch.kraus:
                v=b.apply_batch(state,array(k),(q,)).reshape(count,-1)
                weights.append((v.conj()*v).real.sum(axis=1))
            weights=b.clamp_min(b.stack(weights,axis=1),0)
            totals=b.clamp_min(weights.sum(axis=1),1e-30)
            cumulative=b.cumulative(weights/totals[:,None],1)
            choices=(u[:,None]>=cumulative[:,:-1]).sum(axis=1)
            out=b.zeros(state.shape)
            for index,k in enumerate(ch.kraus):
                v=b.apply_batch(state,array(k),(q,))
                scales=b.where(choices==index,1/b.clamp_min(weights[:,index],1e-30)**.5,0)
                out+=v*scales.reshape((count,)+(1,)*n)
            return out

        def evolve_batch(count):
            state=b.expand_state(initial_vector,count).reshape((count,)+(2,)*n)
            for op in circuit.operations:
                if op.kind=='noise':
                    state=batch_channel(state,op.data,op.wires[0])
                    continue
                if id(op.data) in cx_data:
                    state=b.apply_cx(state,op.wires[0]+1,op.wires[1]+1)
                else:
                    state=b.apply_batch(state,array(op.data,op.kind=='permutation'),op.wires,op.kind=='diagonal',op.kind=='permutation')
                if noise:
                    for ch in noise.channels_for(op.name):
                        for q in op.wires: state=batch_channel(state,ch,q)
            return state

        b.sync()
        started = time.perf_counter()
        cpu_started = time.process_time()
        initial_vector=b.array(psi)
        saved_psi = saved_rho = None
        if method == "trajectory":
            device_probs = b.real_zeros(size)
            # လမ်းကြောင်းတစ်ခုချင်းစီကို သီးခြား simulation လုပ်ပြီး ပျမ်းမျှယူပါ။
            for offset in range(0,trajectories,effective_batch):
                count=min(effective_batch,trajectories-offset)
                state=evolve_batch(count).reshape(count,size)
                device_probs+=(state.conj()*state).real.sum(axis=0)
            probs=b.numpy(device_probs)/trajectories
        elif method == "density_matrix":
            state = evolve(True).reshape(size, size)
            diag = state.diagonal()
            probs = b.numpy(diag.real).copy()
            if retain_state: saved_rho = b.numpy(state)
        else:
            state = evolve(False).reshape(size)
            probs = b.numpy((state.conj()*state).real).copy()
            if retain_state: saved_psi = b.numpy(state)
        b.sync()
        elapsed = time.perf_counter() - started
        cpu_seconds = time.process_time() - cpu_started
        if not np.isfinite(probs).all() or np.min(probs) < -1e-5 or abs(probs.sum()-1) > 1e-3:
            raise RuntimeError("Probability normalization failed")
        probs = np.maximum(np.asarray(probs,dtype=float), 0); probs /= probs.sum()
        # Readout errors affect observed probabilities/counts, never the quantum state.
        if readout_error:
            p = probs.reshape((2,)*n)
            for q in range(n): p = (1-readout_error)*p + readout_error*np.flip(p, axis=q)
            probs = p.reshape(-1)
        counts = dict(sorted(Counter(format(int(i), f"0{n}b")
                         for i in rng.choice(size, size=shots, p=probs)).items())) if shots else {}
        metadata = dict(n_qubits=n, method=method, backend=b.name, device=b.device,
                        precision=self.precision, seed=seed, shots=int(shots),
                        trajectories=trajectories if method == "trajectory" else None,
                        simulation_seconds=elapsed, process_cpu_seconds=cpu_seconds,
                        state_storage_bytes=base_bytes, estimated_working_bytes=working_bytes,
                        readout_error=readout_error, operations=len(circuit.operations),
                        batch_size=effective_batch,compiled=prepared,compile_seconds=compile_seconds,
                        gate_cache_bytes=self._cache_bytes,
                        random_engine=random_engine,noise_fastpath=bool(noise_fastpath),
                        local_gates=b.local_gates,
                        trajectory_sampling_device=b.device if method=='trajectory' else None,
                        probability_kind="trajectory_estimate" if method=="trajectory" else "exact")
        if b.torch and b.device.startswith("cuda"):
            metadata["cuda_allocated_bytes_at_end"] = b.torch.cuda.memory_allocated(b.device)
        return Result(probs, counts, metadata, saved_psi, saved_rho)
