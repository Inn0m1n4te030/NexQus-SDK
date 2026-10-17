"""Small, deliberately slow full-matrix oracle, independent of every SDK.

Only used for correctness at <=5 qubits; excluded from performance timings.
Bit-index embedding is intentionally unlike NexQus tensor kernels.
"""
import numpy as np


def kraus(gate, p):
    if gate == 'amplitude_damping':
        return [np.diag([1,np.sqrt(1-p)]), np.array([[0,np.sqrt(p)],[0,0]])]
    if gate == 'depolarizing':
        # E(rho)=(1-p)rho+p I/2. This is not PennyLane's p/3 convention.
        return [np.sqrt(1-3*p/4)*np.eye(2), np.sqrt(p/4)*np.array([[0,1],[1,0]]),
                np.sqrt(p/4)*np.array([[0,-1j],[1j,0]]), np.sqrt(p/4)*np.diag([1,-1])]
    raise ValueError('Unknown noise channel')


def matrix(op):
    name, angle = op['gate'], op.get('angle', 0)
    if name == 'H': return np.array([[1,1],[1,-1]],complex)/np.sqrt(2)
    if name == 'RX': return np.array([[np.cos(angle/2),-1j*np.sin(angle/2)],[-1j*np.sin(angle/2),np.cos(angle/2)]])
    if name == 'RY': return np.array([[np.cos(angle/2),-np.sin(angle/2)],[np.sin(angle/2),np.cos(angle/2)]],complex)
    if name == 'RZ': return np.diag(np.exp(np.array([-1j,1j])*angle/2))
    if name == 'CP': return np.diag([1,1,1,np.exp(1j*angle)])
    if name == 'CZ': return np.diag([1,1,1,-1]).astype(complex)
    if name == 'CX': return np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]],complex)
    if name == 'SWAP': return np.eye(4,dtype=complex)[[0,2,1,3]]
    raise ValueError('Unknown gate')


def embed(local, wires, n):
    full = np.zeros((2**n,2**n),complex)
    masks = [1 << (n-1-wire) for wire in wires]
    for column in range(2**n):
        local_column = 0
        for mask in masks: local_column = local_column*2 + bool(column & mask)
        for local_row in range(2**len(wires)):
            row = column
            for index, mask in enumerate(masks):
                if local_row & (1 << (len(wires)-1-index)): row |= mask
                else: row &= ~mask
            full[row,column] = local[local_row,local_column]
    return full


def dense_reference(spec):
    n = spec['n_qubits']
    if n > 5: raise ValueError('Dense oracle is limited to five qubits')
    state = np.zeros(2**n,complex); state[0] = 1
    density = spec['method'] == 'density_matrix'
    if density: state = np.outer(state,state.conj())
    for op in spec['operations']:
        if op['gate'] in ('amplitude_damping','depolarizing'):
            if not density: raise ValueError('Noise needs density_matrix')
            matrices = [embed(k, op['wires'], n) for k in kraus(op['gate'],op['angle'])]
            state = sum(k @ state @ k.conj().T for k in matrices)
        else:
            u = embed(matrix(op),op['wires'],n)
            state = u @ state @ u.conj().T if density else u @ state
    return state


def accuracy(reference, state, method, precision, probabilities=None):
    tolerance = 2e-5 if precision == 'single' else 1e-9
    if state.shape != reference.shape or not np.isfinite(state).all():
        return dict(passed=False, reason='Invalid state shape or nonfinite entries')
    density = method == 'density_matrix'
    p = state.diagonal().real if density else abs(state)**2
    expected_p = reference.diagonal().real if density else abs(reference)**2
    probability_error = 0.0
    if probabilities is not None:
        if probabilities.shape != p.shape or not np.isfinite(probabilities).all():
            return dict(passed=False, reason='Invalid probability shape or nonfinite entries')
        probability_error = float(np.max(abs(probabilities-p)))
    if density:
        aligned = state
        hermitian = float(np.max(abs(state-state.conj().T)))
        normalization = float(abs(np.trace(state)-1))
        minimum_eigenvalue = float(np.linalg.eigvalsh(state).min()) if state.shape[0] <= 32 else None
    else:
        overlap = np.vdot(reference,state)
        aligned = state * (overlap.conjugate()/abs(overlap) if abs(overlap) else 1)
        hermitian, minimum_eigenvalue = None, None
        normalization = float(abs(np.vdot(state,state).real-1))
    error = float(np.max(abs(aligned-reference)))
    tvd = float(.5*np.sum(abs(p-expected_p)))
    passed = (error <= tolerance and tvd <= 10*tolerance and normalization <= 10*tolerance
              and probability_error <= tolerance
              and p.min() >= -tolerance and (hermitian is None or hermitian <= tolerance)
              and (minimum_eigenvalue is None or minimum_eigenvalue >= -tolerance))
    return dict(passed=bool(passed), max_state_error=error, probability_tvd=tvd,
                probability_state_error=probability_error,
                normalization_error=normalization, hermiticity_error=hermitian,
                minimum_eigenvalue=minimum_eigenvalue, tolerance=tolerance)
