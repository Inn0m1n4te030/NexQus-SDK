"""Statevector, density-matrix and Monte Carlo trajectory simulation."""
from collections import Counter
from dataclasses import dataclass
import time
import numpy as np
from .backend import Backend
from .circuit import Circuit, positive_integer, I, X, Y, Z
from .noise import NoiseModel


@dataclass
class Result:
    probabilities: np.ndarray
    counts: dict
    metadata: dict
    statevector: object = None
    density_matrix: object = None

    @property
    def most_likely(self):
        return format(int(self.probabilities.argmax()), f"0{self.metadata['n_qubits']}b")

    def marginal(self, wires):
        n = self.metadata["n_qubits"]
        wires = tuple(wires)
        Circuit(n)._wires(wires)
        axes = wires + tuple(q for q in range(n) if q not in wires)
        return self.probabilities.reshape((2,)*n).transpose(axes).reshape(2**len(wires), -1).sum(axis=1)

    def expectation(self, pauli):
        """Pauli string in q0..qn order, e.g. 'ZI' or 'XX'.

        Requires retain_state=True. Trajectory runs do not retain a mixed state.
        """
        n = self.metadata["n_qubits"]
        if len(pauli) != n or any(p not in "IXYZ" for p in pauli):
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
    method='trajectory': finite-sample noisy ensemble, one trajectory at a time.
    max_memory_mb limits estimated core working storage, not process RAM.
    """
    def __init__(self, backend="numpy", device="cpu", method="auto", precision="double",
                 max_memory_mb=1024):
        if method not in ("auto", "statevector", "density_matrix", "trajectory"):
            raise ValueError("Unknown simulation method")
        if not np.isfinite(max_memory_mb) or max_memory_mb <= 0:
            raise ValueError("max_memory_mb must be positive")
        self.backend = Backend(backend, device, precision)
        self.method = method
        self.precision = precision
        self.max_memory_mb = float(max_memory_mb)

    def run(self, circuit, shots=1024, seed=None, noise=None, trajectories=256,
            initial_state=None, retain_state=True, readout_error=0.0):
        if not isinstance(circuit, Circuit): raise TypeError("Expected a Circuit")
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
        base_bytes = self.backend.itemsize * (size**2 if method == "density_matrix" else size)
        # Working tensor, result, transpose/matmul temporaries; not a hard allocator cap.
        working_bytes = base_bytes * 6
        if working_bytes > self.max_memory_mb * 1024**2:
            raise MemoryError(f"Estimated working tensors need {working_bytes/1024**2:.1f} MiB; "
                              "increase max_memory_mb or use fewer qubits/trajectory mode")
        if initial_state is None:
            psi = np.zeros(size, complex); psi[0] = 1
        else:
            psi = np.asarray(initial_state, dtype=complex)
            if psi.shape != (size,) or not np.isfinite(psi).all() or not np.isclose(np.vdot(psi,psi).real,1):
                raise ValueError("initial_state must be a finite normalized statevector")
        rng = np.random.default_rng(seed)
        b = self.backend
        cache = {}
        def array(data):
            key = id(data)
            if key not in cache: cache[key] = b.array(data)
            return cache[key]

        def transform(state, data, wires, density=False, diagonal=False):
            state = b.apply(state, array(data), wires, diagonal)
            if density:
                state = b.apply(state, array(data).conj(), tuple(n+q for q in wires), diagonal)
            return state

        def channel(state, ch, q, density):
            if density:
                total = b.zeros(state.shape)
                for k in ch.kraus: total = total + transform(state, k, (q,), True)
                return total
            # Two passes keep trajectory temporary storage independent of Kraus rank.
            weights = []
            for k in ch.kraus:
                v = b.apply(state, array(k), (q,))
                weights.append(float(b.numpy((v.conj()*v).real.sum())))
            weights = np.maximum(weights, 0)
            total = weights.sum()
            if total <= 0: raise RuntimeError("Invalid zero-probability quantum trajectory")
            idx = int(rng.choice(len(weights), p=weights/total))
            v = b.apply(state, array(ch.kraus[idx]), (q,))
            return v / np.sqrt(weights[idx])

        def evolve(density):
            start = np.outer(psi, psi.conj()) if density else psi
            state = b.array(start).reshape((2,)*(2*n if density else n))
            for op in circuit.operations:
                if op.kind == "noise":
                    state = channel(state, op.data, op.wires[0], density)
                    continue
                state = transform(state, op.data, op.wires, density, op.kind == "diagonal")
                if noise:
                    for ch in noise.channels_for(op.name):
                        for q in op.wires: state = channel(state, ch, q, density)
            return state

        b.sync()
        started = time.perf_counter()
        cpu_started = time.process_time()
        saved_psi = saved_rho = None
        if method == "trajectory":
            trajectories = positive_integer(trajectories, "trajectories")
            probs = np.zeros(size, float)
            # လမ်းကြောင်းတစ်ခုချင်းစီကို သီးခြား simulation လုပ်ပြီး ပျမ်းမျှယူပါ။
            for _ in range(trajectories):
                state = evolve(False).reshape(size)
                probs += b.numpy((state.conj()*state).real)
            probs /= trajectories
        elif method == "density_matrix":
            state = evolve(True).reshape(size, size)
            diag = state.diagonal()
            probs = b.numpy(diag.real).copy()
            if retain_state: saved_rho = b.numpy(state).copy()
        else:
            state = evolve(False).reshape(size)
            probs = b.numpy((state.conj()*state).real).copy()
            if retain_state: saved_psi = b.numpy(state).copy()
        b.sync()
        elapsed = time.perf_counter() - started
        cpu_seconds = time.process_time() - cpu_started
        if not np.isfinite(probs).all() or np.min(probs) < -1e-5 or abs(probs.sum()-1) > 1e-3:
            raise RuntimeError("Probability normalization failed")
        probs = np.maximum(probs, 0); probs /= probs.sum()
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
                        probability_kind="trajectory_estimate" if method=="trajectory" else "exact")
        if b.torch and b.device.startswith("cuda"):
            metadata["cuda_allocated_bytes_at_end"] = b.torch.cuda.memory_allocated(b.device)
        return Result(probs, counts, metadata, saved_psi, saved_rho)
