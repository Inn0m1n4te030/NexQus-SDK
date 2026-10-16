"""Circuit construction. q0 is the LEFTMOST (most significant) bit."""
from dataclasses import dataclass
from functools import wraps
import numbers
import numpy as np
from .parameters import Expression, bindings_dict, real_scalar

I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.diag([1, -1]).astype(complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)


def positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


@dataclass(frozen=True)
class Operation:
    name: str
    wires: tuple
    data: object
    kind: str = "unitary"
    condition: object = None


class Circuit:
    """Chain gates or compose circuits using ordinary Python functions."""
    def __init__(self, n_qubits):
        self.n_qubits = positive_integer(n_qubits, "n_qubits")
        self.operations = []

    def _wires(self, wires):
        if isinstance(wires, numbers.Integral) and not isinstance(wires, bool):
            wires = (wires,)
        else:
            try:
                wires = tuple(wires)
            except TypeError as exc:
                raise ValueError("Provide wire indices as integers") from exc
        if not wires or len(set(wires)) != len(wires):
            raise ValueError("Provide distinct nonempty wires")
        if any(isinstance(q, bool) or not isinstance(q, numbers.Integral)
               or q < 0 or q >= self.n_qubits for q in wires):
            raise ValueError("Wire index out of range")
        return tuple(int(q) for q in wires)

    def unitary(self, matrix, wires, name="U"):
        wires = self._wires(wires)
        u = np.array(matrix, dtype=complex, copy=True)
        d = 2 ** len(wires)
        if u.shape != (d, d) or not np.isfinite(u).all():
            raise ValueError("Unitary shape must be (2**len(wires), 2**len(wires))")
        if not np.allclose(u.conj().T @ u, np.eye(d), atol=1e-10, rtol=1e-10):
            raise ValueError("Matrix is not unitary")
        u.flags.writeable = False
        self.operations.append(Operation(name, wires, u))
        return self

    def diagonal(self, diagonal, wires, name="diagonal"):
        wires = self._wires(wires)
        d = np.array(diagonal, dtype=complex, copy=True)
        if (d.shape != (2 ** len(wires),) or not np.isfinite(d).all()
                or not np.allclose(abs(d), 1, atol=1e-10, rtol=1e-10)):
            raise ValueError("Diagonal must contain one unit-modulus entry per basis state")
        d.flags.writeable = False
        self.operations.append(Operation(name, wires, d, "diagonal"))
        return self

    def h(self, q): return self.unitary(H, q, "H")
    def x(self, q): return self.unitary(X, q, "X")
    def y(self, q): return self.unitary(Y, q, "Y")
    def z(self, q): return self.unitary(Z, q, "Z")
    def s(self, q): return self.phase(np.pi / 2, q)
    def t(self, q): return self.phase(np.pi / 4, q)

    def phase(self, theta, q):
        if isinstance(theta,Expression): return self._parameter_gate('phase',theta,q)
        theta=real_scalar(theta)
        return self.diagonal([1, np.exp(1j * theta)], q, "P")

    def _rotation(self, theta, q, pauli, name):
        if isinstance(theta,Expression): return self._parameter_gate(name.lower(),theta,q)
        theta=real_scalar(theta)
        u = np.cos(theta / 2) * I - 1j * np.sin(theta / 2) * pauli
        return self.unitary(u, q, name)

    def rx(self, theta, q): return self._rotation(theta, q, X, "RX")
    def ry(self, theta, q): return self._rotation(theta, q, Y, "RY")
    def rz(self, theta, q): return self._rotation(theta, q, Z, "RZ")

    def controlled(self, matrix, control, target, name="controlled-U"):
        u = np.asarray(matrix, dtype=complex)
        if u.shape != (2, 2):
            raise ValueError("controlled() currently accepts a single-qubit target matrix")
        full = np.eye(4, dtype=complex)
        full[2:, 2:] = u
        return self.unitary(full, (control, target), name)

    def cx(self, control, target): return self.controlled(X, control, target, "CX")
    def cz(self, control, target): return self.diagonal([1, 1, 1, -1], (control, target), "CZ")
    def cp(self, theta, control, target):
        if isinstance(theta,Expression): return self._parameter_gate('cp',theta,(control,target))
        theta=real_scalar(theta)
        return self.diagonal([1, 1, 1, np.exp(1j * theta)], (control, target), "CP")

    def _parameter_gate(self, gate, expression, wires):
        wires=self._wires(wires)
        expected=2 if gate=='cp' else 1
        if len(wires)!=expected: raise ValueError('Incorrect number of gate wires')
        name='P' if gate=='phase' else gate.upper()
        self.operations.append(Operation(name,wires,(gate,expression),'parameterized'))
        return self

    @property
    def parameters(self):
        names=set()
        for op in self.operations:
            if op.kind=='parameterized': names.update(op.data[1].names)
        return tuple(sorted(names))

    def bind(self, values):
        """Return a new concrete circuit; missing/unknown parameter names are errors."""
        values=bindings_dict(values)
        required=set(self.parameters)
        if set(values)!=required:
            raise ValueError(f'Missing bindings: {sorted(required-set(values))}; '
                             f'unknown bindings: {sorted(set(values)-required)}')
        out=Circuit(self.n_qubits)
        for op in self.operations:
            if op.kind=='parameterized':
                gate,expression=op.data
                getattr(out,gate)(expression.evaluate(values),*op.wires)
                concrete=out.operations[-1]
                out.operations[-1]=Operation(op.name,concrete.wires,concrete.data,concrete.kind,op.condition)
            else: out.operations.append(op)
        return out

    def swap(self, a, b):
        return self.unitary([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], (a,b), "SWAP")

    def noise(self, channel, q):
        from .noise import Noise
        if not isinstance(channel, Noise):
            raise TypeError("channel must be a Noise")
        wires = self._wires(q)
        if len(wires) != 1:
            raise ValueError("Noise channels currently act on one qubit")
        self.operations.append(Operation(channel.name, wires, channel, "noise"))
        return self

    @property
    def classical_keys(self):
        return tuple(dict.fromkeys(op.data for op in self.operations if op.kind=='measure'))

    @property
    def dynamic(self):
        return any(op.kind in ('measure','reset') or op.condition is not None for op in self.operations)

    def measure(self,q,key):
        """Projectively measure one qubit in Z and overwrite a named classical bit."""
        wires=self._wires(q)
        if len(wires)!=1: raise ValueError('measure() acts on one qubit at a time')
        if not isinstance(key,str) or not key.strip(): raise ValueError('Classical key must be a nonempty string')
        self.operations.append(Operation('MEASURE',wires,key,'measure'))
        return self

    def reset(self,q):
        """Reset one qubit to |0>, discarding its previous value."""
        wires=self._wires(q)
        if len(wires)!=1: raise ValueError('reset() acts on one qubit at a time')
        self.operations.append(Operation('RESET',wires,None,'reset'))
        return self

    def c_if(self,key,value=1):
        """Condition the immediately preceding gate/reset/noise on a measured bit."""
        if not self.operations or self.operations[-1].kind not in ('unitary','diagonal','parameterized','reset','noise'):
            raise ValueError('c_if() must follow a gate, reset, or noise operation')
        if not isinstance(key,str) or key not in tuple(op.data for op in self.operations[:-1] if op.kind=='measure'):
            raise ValueError('Condition requires a classical bit measured earlier in the circuit')
        if isinstance(value,bool) or not isinstance(value,numbers.Integral) or value not in (0,1):
            raise ValueError('Condition value must be integer 0 or 1')
        op=self.operations[-1]
        if op.condition is not None: raise ValueError('Operation already has a condition')
        self.operations[-1]=Operation(op.name,op.wires,op.data,op.kind,(key,int(value)))
        return self

    def compose(self, other, wires=None, classical_map=None):
        if not isinstance(other, Circuit):
            raise TypeError("Expected Circuit")
        mapped = self._wires(tuple(range(self.n_qubits)) if wires is None else wires)
        if len(mapped) != other.n_qubits:
            raise ValueError("Mapping length must match the source circuit")
        # Snapshot allows c.compose(c) without growing the iteration indefinitely.
        classical_map={} if classical_map is None else dict(classical_map)
        if set(classical_map)-set(other.classical_keys): raise ValueError('Unknown classical key in mapping')
        if any(not isinstance(v,str) or not v.strip() for v in classical_map.values()):
            raise ValueError('Mapped classical keys must be nonempty strings')
        translated=[classical_map.get(key,key) for key in other.classical_keys]
        if len(set(translated))!=len(translated): raise ValueError('Classical mapping must not merge distinct keys')
        for op in list(other.operations):
            data=classical_map.get(op.data,op.data) if op.kind=='measure' else op.data
            condition=None if op.condition is None else (classical_map.get(op.condition[0],op.condition[0]),op.condition[1])
            self.operations.append(Operation(op.name,tuple(mapped[q] for q in op.wires),data,op.kind,condition))
        return self

    def inverse(self):
        if self.dynamic or any(op.kind == "noise" for op in self.operations):
            raise ValueError("A noisy or dynamic circuit has no unitary inverse")
        out = Circuit(self.n_qubits)
        for op in reversed(self.operations):
            if op.kind == 'parameterized':
                gate,expression=op.data
                out.operations.append(Operation(op.name+'†',op.wires,(gate,-expression),op.kind))
                continue
            data = op.data.conj() if op.kind == "diagonal" else op.data.conj().T
            out.operations.append(Operation(op.name + "†", op.wires, data, op.kind))
        return out

    def __repr__(self):
        return f"Circuit(n_qubits={self.n_qubits}, operations={len(self.operations)})"


def circuit(n_qubits):
    """@circuit(2) turns def prepare(q, theta): ... into a Circuit factory."""
    positive_integer(n_qubits, "n_qubits")
    def decorator(fn):
        @wraps(fn)
        def build(*args, **kwargs):
            q = Circuit(n_qubits)
            fn(q, *args, **kwargs)
            return q
        return build
    return decorator
