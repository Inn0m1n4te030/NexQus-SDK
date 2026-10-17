"""Optional Amazon Braket bridge. Importing/preparing never contacts AWS.

Only static, bound circuits with recognized one/two-qubit gates are translated.
Cloud submission is explicit; no retry or automatic device selection is done.
"""
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import numbers
import re

import numpy as np

from .circuit import Circuit, I, X, Y, Z, H, positive_integer


def _sdk():
    try:
        from braket.circuits import Circuit as NativeCircuit
    except ImportError as exc:
        raise ImportError("Install nexqus-sdk[braket] with Python 3.11+ for Braket execution") from exc
    return NativeCircuit


def _money(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite nonnegative USD amount")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"{name} must be a finite nonnegative USD amount") from exc
    if not result.is_finite() or result < 0:
        raise ValueError(f"{name} must be a finite nonnegative USD amount")
    return result


def _recognized_gate(op):
    # Never trust operation names or display parameters as execution semantics.
    if op.kind not in ('unitary', 'diagonal') or len(op.wires) > 2:
        raise ValueError(f"Unsupported Braket operation {op.name!r} ({op.kind}); decompose it first")
    size = 2 ** len(op.wires)
    data = np.asarray(op.data, dtype=complex)
    if op.kind == 'diagonal':
        if data.shape != (size,):
            raise ValueError('Invalid diagonal shape')
        matrix = np.diag(data)
    else:
        matrix = data
    if matrix.shape != (size, size) or not np.isfinite(matrix).all():
        raise ValueError('Invalid gate matrix')
    cx = np.eye(4, dtype=complex); cx[2:, 2:] = X
    swap = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], complex)
    fixed = [('i', I), ('h', H), ('x', X), ('y', Y), ('z', Z),
             ('cnot', cx), ('cz', np.diag([1,1,1,-1])), ('swap', swap)]
    for name, candidate in fixed:
        if matrix.shape == candidate.shape and np.allclose(matrix, candidate, atol=1e-12, rtol=0):
            return name, ()
    if len(op.params) == 1:
        value = op.params[0]
        if isinstance(value, numbers.Real) and not isinstance(value, bool) and np.isfinite(value):
            # Inverse/bound-inverse operations retain the original display angle.
            for angle in (float(value), -float(value)):
                if size == 2:
                    candidates = [(name, np.cos(angle/2)*I - 1j*np.sin(angle/2)*p)
                                  for name, p in [('rx', X), ('ry', Y), ('rz', Z)]]
                    candidates.append(('phaseshift', np.diag([1, np.exp(1j*angle)])))
                else:
                    candidates = [('cphaseshift', np.diag([1,1,1,np.exp(1j*angle)]))]
                for name, candidate in candidates:
                    if np.allclose(matrix, candidate, atol=1e-12, rtol=0):
                        return name, (angle,)
    raise ValueError(f"No supported native Braket translation for {op.name!r}; decompose it first")


@dataclass(frozen=True)
class BraketProgram:
    """Immutable execution plan from prepare_braket; no AWS objects or credentials."""
    n_qubits: int
    shots: int
    instructions: tuple

    def __post_init__(self):
        n = positive_integer(self.n_qubits, 'n_qubits')
        shots = positive_integer(self.shots, 'shots')
        arities = {'i':(1,0), 'h':(1,0), 'x':(1,0), 'y':(1,0), 'z':(1,0),
                   'rx':(1,1), 'ry':(1,1), 'rz':(1,1), 'phaseshift':(1,1),
                   'cnot':(2,0), 'cz':(2,0), 'swap':(2,0), 'cphaseshift':(2,1)}
        instructions = []
        for name, wires, angles in self.instructions:
            if name not in arities:
                raise ValueError(f'Unsupported native Braket gate: {name!r}')
            wires = Circuit(n)._wires(wires)
            angles = tuple(angles)
            if (len(wires),len(angles)) != arities[name]:
                raise ValueError('Wrong gate wire/angle count')
            if any(isinstance(a,bool) or not isinstance(a,numbers.Real) or not np.isfinite(a) for a in angles):
                raise ValueError('Gate angles must be finite real numbers')
            instructions.append((name, wires, tuple(float(a) for a in angles)))
        used = {q for _, wires, _ in instructions for q in wires}
        if used != set(range(n)):
            raise ValueError('Every logical wire must appear; use prepare_braket to include idle wires')
        object.__setattr__(self, 'n_qubits', n)
        object.__setattr__(self, 'shots', shots)
        object.__setattr__(self, 'instructions', tuple(instructions))

    @property
    def gate_counts(self):
        return dict(Counter(name for name, _, _ in self.instructions))

    def to_braket(self):
        """Create a fresh native circuit locally. Does not create an AWS session."""
        native = _sdk()()
        for name, wires, angles in self.instructions:
            getattr(native, name)(*wires, *angles)
        return native

    def estimate_cost(self, *, task_price_usd, shot_price_usd):
        """QPU per-task + per-shot estimate using caller-supplied current prices.

        Not a live quote or billing cap; excludes tax, S3, reservations and extras.
        Not applicable to cloud simulators billed by duration.
        """
        return (_money(task_price_usd, 'task_price_usd')
                + self.shots * _money(shot_price_usd, 'shot_price_usd'))

    def validate_device(self, capabilities):
        """Validate a Braket capability dict offline; routing stays with Braket.

        Requires OpenQASM, declared gate support, qubit capacity and shot range.
        Logical wire connectivity is not assumed to be physical connectivity.
        """
        try:
            action = capabilities['action']['braket.ir.openqasm.program']
            supported = {x.lower() for x in action['supportedOperations']}
            capacity = capabilities['paradigm']['qubitCount']
            low, high = capabilities['service']['shotsRange']
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise ValueError('Missing gate-model OpenQASM capability information') from exc
        if (any(isinstance(v,bool) or not isinstance(v,numbers.Integral) for v in (capacity,low,high))
                or capacity < 1 or low < 0 or high < low):
            raise ValueError('Invalid device qubit capacity or shot range')
        if self.n_qubits > capacity:
            raise ValueError(f'Circuit needs {self.n_qubits} qubits; device has {capacity}')
        if not low <= self.shots <= high:
            raise ValueError(f'Shots must be in the device range {low}..{high}')
        missing = set(self.gate_counts) - supported
        if missing:
            raise ValueError(f'Device does not declare support for gates: {sorted(missing)}')
        return self

    def run_local(self, *, max_qubits=20):
        """Run Braket's local CPU simulator, with no AWS account or cloud task."""
        max_qubits = positive_integer(max_qubits, 'max_qubits')
        if self.n_qubits > max_qubits:
            raise ValueError('Local simulation exceeds max_qubits; explicitly raise it if memory permits')
        native = self.to_braket()
        from braket.devices import LocalSimulator
        result = LocalSimulator().run(native, shots=self.shots).result()
        return BraketResult.from_native(result, self.n_qubits, self.shots,
                                        metadata={'backend': 'braket_local'})


def prepare_braket(circuit, *, shots=1000, bindings=None):
    """Offline validation/translation; measures every logical wire at the end.

    Rejects noise, measurement/reset/feed-forward, permutations and unknown
    custom operations. Idle wires receive I to preserve q0-first bitstrings.
    """
    if not isinstance(circuit, Circuit):
        raise TypeError('Expected a NexQus Circuit')
    shots = positive_integer(shots, 'shots')
    if bindings is not None:
        circuit = circuit.bind(bindings)
    if circuit.parameters:
        raise ValueError('Bind every circuit parameter before preparing Braket execution')
    instructions = []
    used = set()
    for op in tuple(circuit.operations):
        wires = circuit._wires(op.wires)
        if op.condition is not None:
            raise ValueError('Braket adapter does not support classical feed-forward')
        name, angles = _recognized_gate(op)
        instructions.append((name, wires, angles)); used.update(wires)
    instructions.extend(('i', (wire,), ()) for wire in range(circuit.n_qubits) if wire not in used)
    return BraketProgram(circuit.n_qubits, shots, tuple(instructions))


@dataclass
class BraketResult:
    """Sparse empirical measurements in NexQus q0-first order; no retained state."""
    counts: dict
    n_qubits: int
    shots: int
    metadata: dict

    @classmethod
    def from_native(cls, result, n_qubits, shots, *, metadata=None):
        n_qubits = positive_integer(n_qubits, 'n_qubits')
        shots = positive_integer(shots, 'shots')
        measured = tuple(result.measured_qubits)
        if (len(measured) != n_qubits or any(isinstance(q, bool) or not isinstance(q, numbers.Integral)
                                           for q in measured)
                or set(measured) != set(range(n_qubits))):
            raise ValueError('Result must measure every logical qubit exactly once')
        columns = tuple(measured.index(q) for q in range(n_qubits))
        counts = Counter()
        for bits, count in result.measurement_counts.items():
            if (not isinstance(bits, str) or len(bits) != n_qubits or set(bits)-{'0','1'}
                    or isinstance(count, bool) or not isinstance(count, numbers.Integral) or count < 0):
                raise ValueError('Malformed Braket measurement counts')
            if count:
                counts[''.join(bits[i] for i in columns)] += int(count)
        if sum(counts.values()) != shots:
            raise ValueError('Measurement count total differs from requested shots')
        native_shots = getattr(getattr(result, 'task_metadata', None), 'shots', shots)
        if native_shots != shots:
            raise ValueError('Braket task shot metadata differs from requested shots')
        info = dict(metadata or {}, n_qubits=n_qubits, shots=shots,
                    probability_kind='empirical', measured_qubits=measured)
        return cls(dict(sorted(counts.items())), n_qubits, shots, info)

    @property
    def probabilities(self):
        return {bits: count/self.shots for bits, count in self.counts.items()}

    @property
    def most_likely(self):
        return max(sorted(self.counts), key=self.counts.get)

    def expectation(self, pauli):
        """Estimate an I/Z Pauli string from computational-basis measurements."""
        if not isinstance(pauli, str) or len(pauli) != self.n_qubits or set(pauli)-{'I','Z'}:
            raise ValueError('Sampled Z-basis results support only one I/Z per qubit; rotate for X/Y')
        return sum(count * (-1 if sum(bit == '1' and p == 'Z' for bit, p in zip(bits, pauli)) % 2
                            else 1) for bits, count in self.counts.items()) / self.shots

    def to_result(self, *, max_qubits=20):
        """Explicitly allocate a dense NexQus Result, capped by max_qubits."""
        from .simulator import Result
        max_qubits = positive_integer(max_qubits, 'max_qubits')
        if self.n_qubits > max_qubits:
            raise ValueError('Dense result exceeds max_qubits; keep the sparse BraketResult')
        probabilities = np.zeros(2**self.n_qubits)
        for bits, count in self.counts.items():
            probabilities[int(bits, 2)] = count/self.shots
        return Result(probabilities, dict(self.counts), dict(self.metadata))


class BraketJob:
    """Wrap one submitted/existing task. Polling never submits another task."""
    def __init__(self, task, n_qubits, shots, device_arn):
        self._task = task
        self.n_qubits = positive_integer(n_qubits, 'n_qubits')
        self.shots = positive_integer(shots, 'shots')
        self.device_arn = device_arn

    @property
    def id(self):
        return self._task.id

    def state(self):
        return self._task.state()

    def cancel(self):
        """Request cancellation. AWS may already have executed/billed the task."""
        return self._task.cancel()

    def result(self):
        native = self._task.result()
        if native is None:
            raise RuntimeError(f'No result for task {self.id}; inspect state(), then poll again if needed')
        return BraketResult.from_native(native, self.n_qubits, self.shots,
            metadata={'backend': 'amazon_braket', 'device_arn': self.device_arn, 'task_arn': self.id})


class BraketBackend:
    """Explicit AWS QPU access. Constructor/prepare are offline; submit is paid.

    AWS credentials use the official SDK credential chain or an AwsSession.
    No passwords/keys are accepted, logged or stored by this adapter.
    """
    def __init__(self, device_arn, *, aws_session=None):
        if not isinstance(device_arn, str) or not re.fullmatch(
                r'arn:aws:braket:[a-z0-9-]*::device/(qpu|quantum-simulator)/[A-Za-z0-9_./-]+', device_arn):
            raise ValueError('Expected an Amazon Braket device ARN')
        self.device_arn = device_arn
        self._aws_session = aws_session

    def inspect_device(self):
        """Read current AWS device metadata; network access, but no task creation."""
        _sdk()
        from braket.aws import AwsDevice
        return AwsDevice(self.device_arn, aws_session=self._aws_session)

    def submit(self, program, *, allow_paid=False, task_price_usd=None,
               shot_price_usd=None, max_estimated_cost_usd=None, s3_destination_folder=None):
        """Submit exactly one QPU task after explicit opt-in and price preflight.

        Caller prices must be current for this device. The estimate is not an
        AWS spending limit. Duration-billed cloud simulators are not supported.
        No automatic retries: a connection failure can leave a created task.
        """
        if allow_paid is not True:
            raise PermissionError('Cloud submission is disabled; explicitly pass allow_paid=True to incur charges')
        if not isinstance(program, BraketProgram):
            raise TypeError('Expected a prepared BraketProgram')
        if ':device/qpu/' not in self.device_arn:
            raise ValueError('Cloud simulator time-based pricing is not supported; use run_local()')
        estimate = program.estimate_cost(task_price_usd=task_price_usd, shot_price_usd=shot_price_usd)
        limit = _money(max_estimated_cost_usd, 'max_estimated_cost_usd')
        if estimate > limit:
            raise ValueError(f'Estimated task cost ${estimate} exceeds requested limit ${limit}')
        if s3_destination_folder is not None:
            if (not isinstance(s3_destination_folder, tuple) or len(s3_destination_folder) != 2
                    or not all(isinstance(x, str) for x in s3_destination_folder)
                    or not s3_destination_folder[0]):
                raise ValueError('s3_destination_folder must be (bucket, prefix)')
        # Construct/validate native instructions before any network access.
        native = program.to_braket()
        device = self.inspect_device()
        if device.status != 'ONLINE':
            raise ValueError(f'Device is not ONLINE: {device.status}')
        program.validate_device(device.properties.dict())
        task = device.run(native, shots=program.shots, s3_destination_folder=s3_destination_folder,
                          poll_timeout_seconds=300, poll_interval_seconds=5)
        return BraketJob(task, program.n_qubits, program.shots, self.device_arn)

    def get_job(self, task_arn, *, n_qubits, shots):
        """Reconnect to an existing task (network reads only); never resubmit."""
        n_qubits = positive_integer(n_qubits, 'n_qubits')
        shots = positive_integer(shots, 'shots')
        if not isinstance(task_arn, str) or not re.fullmatch(
                r'arn:aws:braket:[a-z0-9-]+:[0-9]{12}:quantum-task/[A-Za-z0-9-]+', task_arn):
            raise ValueError('Expected an Amazon Braket quantum-task ARN')
        _sdk()
        from braket.aws import AwsQuantumTask
        task = AwsQuantumTask(task_arn, aws_session=self._aws_session,
                              poll_timeout_seconds=300, poll_interval_seconds=5)
        info = task.metadata()
        if info['deviceArn'] != self.device_arn or info['shots'] != shots:
            raise ValueError('Existing task device or shot count differs from requested metadata')
        return BraketJob(task, n_qubits, shots, self.device_arn)
