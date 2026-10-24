"""Explicit walk device selection; host-side analysis remains NumPy."""
from .simulator import resolve_simulator


def walk_simulator(device='cpu', backend=None):
    return resolve_simulator(device=device, backend=backend, precision='double')


def device_scope(simulator):
    b = simulator.backend
    return (f'Quantum circuit backend={b.name}, device={b.device}, complex128. '
            'Graph construction, eigendecomposition, classical baseline, detector '
            'bookkeeping and report analysis run on CPU. State transfers are included. ')
