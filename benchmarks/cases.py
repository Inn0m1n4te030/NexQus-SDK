"""SDK-independent circuit specifications; q0 is the most significant bit."""
import numpy as np


def case(name, n, family, *, layers=3, noise=None, seed=73):
    rng = np.random.default_rng(seed)
    operations = []
    def add(gate, wires, angle=None):
        op = dict(gate=gate, wires=list(wires))
        if angle is not None: op['angle'] = float(angle)
        operations.append(op)
    if family == 'ghz':
        add('H', [0])
        for wire in range(1, n): add('CX', [wire-1, wire])
    elif family == 'qft':
        # Asymmetric input catches wire-order and phase mistakes.
        for wire in range(n): add('RY', [wire], rng.uniform(-2, 2))
        for a in range(n):
            add('H', [a])
            for b in range(a+1, n): add('CP', [b, a], np.pi / 2**(b-a))
        for wire in range(n//2): add('SWAP', [wire, n-wire-1])
    elif family in ('rotations', 'noisy', 'diagonal'):
        for _ in range(layers):
            for wire in range(n):
                if family == 'diagonal':
                    add('H', [wire]); add('RZ', [wire], rng.uniform(-np.pi, np.pi))
                else:
                    for gate in ('RX', 'RY', 'RZ'): add(gate, [wire], rng.uniform(-np.pi, np.pi))
            for a in range(n-1):
                add('CP' if family == 'diagonal' else 'CX', [a+1, a],
                    rng.uniform(-1, 1) if family == 'diagonal' else None)
            if n > 2: add('CZ', [n-1, 0])
            if noise:
                for wire in range(n): add(noise, [wire], .03)
    else: raise ValueError('Unknown circuit family')
    return dict(name=name, n_qubits=n, method='density_matrix' if noise else 'statevector',
                seed=seed, operations=operations, layers=layers, noise=noise,
                noise_placement='after each complete layer, on every wire' if noise else 'none')


def suite(name):
    if name == 'smoke':
        return [case('asymmetric_rotations_3q',3,'rotations',layers=2),
                case('qft_3q',3,'qft'),
                case('amplitude_damping_3q',3,'noisy',layers=2,noise='amplitude_damping'),
                case('depolarizing_3q',3,'noisy',layers=2,noise='depolarizing')]
    if name == 'correctness':
        return [case(f'{noise or family}_{n}q_seed{seed}',n,family,seed=seed,noise=noise)
                for n in (2,3,5) for seed in (7,73,419)
                for family,noise in [('rotations',None),('qft',None),('diagonal',None),
                                     ('noisy','amplitude_damping'),('noisy','depolarizing')]]
    if name == 'standard':
        return [case(f'{family}_{n}q', n, family, layers=6)
                for n in (4,12,18) for family in ('ghz','rotations','qft','diagonal')] + [
                    case(f'{noise}_{n}q',n,'noisy',layers=4,noise=noise)
                    for n in (4,8) for noise in ('amplitude_damping','depolarizing')]
    raise ValueError('Unknown suite')
