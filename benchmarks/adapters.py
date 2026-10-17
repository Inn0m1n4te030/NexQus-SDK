"""Translate the common specification independently into each SDK.

These imports are deliberately lazy. They never enter the nexqus runtime.
Every runner returns a full state and probabilities on CPU, with q0 leftmost.
"""
import numpy as np
from reference import kraus


class Unsupported(ValueError):
    pass


def prepare(engine, spec, precision, threads):
    n, density = spec['n_qubits'], spec['method'] == 'density_matrix'
    if engine == 'nexqus':
        from nexqus import Circuit, Simulator, Noise
        q = Circuit(n)
        for op in spec['operations']:
            name, wires = op['gate'], op['wires']
            if name in ('amplitude_damping','depolarizing'):
                q.noise(getattr(Noise,name)(op['angle']),wires[0])
            elif 'angle' in op: getattr(q,name.lower())(op['angle'],*wires)
            else: getattr(q,name.lower())(*wires)
        sim = Simulator(method=spec['method'], precision=precision)
        compiled = sim.compile(q)
        def run():
            result = compiled.run(shots=0,retain_state=True)
            return (result.density_matrix if density else result.statevector), result.probabilities
        return run, dict(backend='numpy', device='cpu', fusion=False, compiled=True,
                         precision=precision, gate_counts=q.count_ops(), logical_depth=q.depth())
    if engine == 'qiskit':
        from qiskit import QuantumCircuit, transpile
        from qiskit.quantum_info import Kraus
        from qiskit_aer import AerSimulator
        q = QuantumCircuit(n)
        for op in spec['operations']:
            # Physical wire n-1-i makes the returned amplitude order match q0
            # as MSB, including reversed/nonadjacent controls and target wires.
            wires = [n-1-wire for wire in op['wires']]
            name = op['gate']
            if name in ('amplitude_damping','depolarizing'):
                q.append(Kraus(kraus(name,op['angle'])).to_instruction(), wires)
            elif 'angle' in op: getattr(q,name.lower())(op['angle'],*wires)
            else: getattr(q,name.lower())(*wires)
        if density: q.save_density_matrix()
        else: q.save_statevector()
        simulator = AerSimulator(method=spec['method'], device='CPU',precision=precision,
                                 fusion_enable=False,enable_truncation=False,zero_threshold=0,
                                 max_parallel_threads=threads,max_parallel_experiments=1,max_parallel_shots=1)
        q = transpile(q,simulator,optimization_level=0,seed_transpiler=spec['seed'])
        def run():
            result = simulator.run(q,shots=0,seed_simulator=spec['seed']).result()
            if not result.success: raise RuntimeError(str(result.status))
            state = np.asarray(result.data(0)['density_matrix' if density else 'statevector'])
            probabilities = state.diagonal().real.copy() if density else abs(state)**2
            return state, probabilities
        return run, dict(backend='AerSimulator',device='CPU',precision=precision,fusion=False,
                         enable_truncation=False,optimization_level=0,bit_order='logical wire i maps to n-1-i',
                         max_parallel_threads=threads)
    if engine == 'pennylane':
        if precision != 'double':
            raise Unsupported('default.qubit/default.mixed adapter supports complex128 only; no silent promotion')
        import pennylane as qml
        operations = []
        gates = dict(H=qml.Hadamard,RX=qml.RX,RY=qml.RY,RZ=qml.RZ,CX=qml.CNOT,CZ=qml.CZ,
                     CP=qml.ControlledPhaseShift,SWAP=qml.SWAP)
        for op in spec['operations']:
            name, wires = op['gate'], op['wires']
            if name == 'amplitude_damping': operations.append(qml.AmplitudeDamping(op['angle'],wires=wires))
            elif name == 'depolarizing':
                # PL uses (1-lambda)rho + lambda/3*(XrhoX+YrhoY+ZrhoZ).
                operations.append(qml.DepolarizingChannel(3*op['angle']/4,wires=wires))
            elif 'angle' in op: operations.append(gates[name](op['angle'],wires=wires))
            else: operations.append(gates[name](wires=wires))
        device_name = 'default.mixed' if density else 'default.qubit'
        device = qml.device(device_name,wires=n,shots=None)
        tape = qml.tape.QuantumScript(operations,[qml.state()],shots=None)
        def run():
            state = np.asarray(device.execute(tape))
            probabilities = state.diagonal().real.copy() if density else abs(state)**2
            return state, probabilities
        return run, dict(backend=device_name,device='cpu',precision='double',interface=None,
                         shots=None,prebuilt_tape=True,fusion=False)
    raise ValueError('Unknown engine')
