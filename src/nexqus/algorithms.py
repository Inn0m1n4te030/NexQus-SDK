"""Small reusable algorithms. All use the NexQus circuit engine."""
import math
from dataclasses import dataclass
from fractions import Fraction
import numpy as np
from .circuit import Circuit, positive_integer
from .simulator import _initial_vector


def bell():
    """Prepare (|00> + |11>)/sqrt(2)."""
    return Circuit(2).h(0).cx(0,1)


def teleportation():
    """Teleport q0 to q2 using q1/q2 as a Bell resource and ideal feed-forward.

    Start with arbitrary input on q0 and |00> on q1/q2. Classical keys are
    alice_z and alice_x; rename with Circuit.compose(classical_map=...).
    """
    q=Circuit(3).h(1).cx(1,2).cx(0,1).h(0)
    q.measure(0,'alice_z').measure(1,'alice_x')
    return q.x(2).c_if('alice_x',1).z(2).c_if('alice_z',1)


def ghz(n_qubits):
    q = Circuit(n_qubits).h(0)
    for target in range(1,n_qubits): q.cx(0,target)
    return q


def qft(n_qubits, inverse=False):
    """Positive-exponent QFT, q0 most significant, including output swaps."""
    q = Circuit(n_qubits)
    # QFT ကို gate များဖြင့် တည်ဆောက်ထားသဖြင့် dense matrix မလိုအပ်ပါ။
    for j in range(n_qubits):
        q.h(j)
        for k in range(j+1,n_qubits): q.cp(np.pi/2**(k-j), k, j)
    for j in range(n_qubits//2): q.swap(j,n_qubits-1-j)
    return q.inverse() if inverse else q


def grover(n_qubits, marked, iterations=None):
    """Search explicit marked bitstrings. Educational oracle, not a speedup claim.

    Oracle uses a length-2**n diagonal (not a dense matrix). Noise is applied
    after this logical oracle to all wires when a global model is supplied.
    """
    n_qubits = positive_integer(n_qubits,"n_qubits")
    if isinstance(marked, str): marked = [marked]
    marked = list(marked)
    if not marked or any(not isinstance(s,str) or len(s)!=n_qubits or set(s)-set("01") for s in marked):
        raise ValueError("marked must contain nonempty n-bit strings")
    marked = sorted(set(marked))
    if iterations is None:
        theta = math.asin(math.sqrt(len(marked)/2**n_qubits))
        iterations = max(0, round(math.pi/(4*theta)-.5))
    if isinstance(iterations,bool) or not isinstance(iterations,(int,np.integer)) or iterations < 0:
        raise ValueError("iterations must be a nonnegative integer")
    q = Circuit(n_qubits)
    for j in range(n_qubits): q.h(j)
    oracle = np.ones(2**n_qubits,complex)
    for s in marked: oracle[int(s,2)] = -1
    reflection = -np.ones(2**n_qubits,complex); reflection[0] = 1
    for _ in range(iterations):
        q.diagonal(oracle,range(n_qubits),"oracle")
        for j in range(n_qubits): q.h(j)
        q.diagonal(reflection,range(n_qubits),"diffusion_phase")
        for j in range(n_qubits): q.h(j)
    return q


def bernstein_vazirani(secret):
    """Secret bits are returned in the first n wires; final wire is ancilla."""
    if not isinstance(secret,str) or not secret or set(secret)-set("01"):
        raise ValueError("secret must be a nonempty binary string")
    n = len(secret)
    q = Circuit(n+1).x(n).h(n)
    for j in range(n): q.h(j)
    for j,bit in enumerate(secret):
        if bit == "1": q.cx(j,n)
    for j in range(n): q.h(j)
    return q


def deutsch_jozsa(truth_table):
    """Phase oracle for a promised constant/balanced binary truth table.

    No output ancilla is needed; result zero means constant in the ideal case.
    """
    table = np.asarray(truth_table)
    if table.ndim != 1 or len(table)<2 or len(table)&(len(table)-1) or not np.isin(table,[0,1]).all():
        raise ValueError("Provide a binary truth table of length 2**n, n>=1")
    if table.sum() not in (0,len(table)//2,len(table)):
        raise ValueError("Deutsch-Jozsa requires a constant or balanced oracle")
    n = int(math.log2(len(table)))
    q = Circuit(n)
    for j in range(n): q.h(j)
    q.diagonal(1-2*table,range(n),"DJ_oracle")
    for j in range(n): q.h(j)
    return q


def phase_estimation(unitary, counting_qubits=3, target_state=None):
    """QPE for a 2x2 unitary and one target qubit, including inverse QFT.

    For U|psi>=exp(2*pi*i*phase)|psi>, the counting register estimates phase.
    Default target_state is |1>. For a superposition of eigenstates the output
    samples their eigenphases. Precision is counting_qubits bits.
    """
    n = positive_integer(counting_qubits,"counting_qubits")
    u = np.asarray(unitary,dtype=complex)
    Circuit(1).unitary(u,0) # Validate the target unitary.
    psi = np.array([0,1],complex) if target_state is None else _initial_vector(target_state,2)
    prepare = np.column_stack([psi,[-psi[1].conjugate(),psi[0].conjugate()]])
    q = Circuit(n+1).unitary(prepare,n,"prepare_target")
    for j in range(n):
        q.h(j)
        q.controlled(np.linalg.matrix_power(u,2**(n-1-j)),j,n,"controlled_power")
    return q.compose(qft(n,inverse=True),range(n))


def qaoa_maxcut(n_qubits, edges, gammas, betas):
    """Build QAOA ansatz for unweighted MaxCut; caller optimizes angles.

    Cost C=sum((I-ZuZv)/2); mixer exp(-i beta sum X).
    """
    q = Circuit(n_qubits)
    gammas,betas = tuple(gammas),tuple(betas)
    if not gammas or len(gammas)!=len(betas): raise ValueError("Provide matching nonempty angle lists")
    edges = tuple(tuple(edge) for edge in edges)
    for edge in edges:
        if len(edge)!=2: raise ValueError("Each edge must have two wires")
        q._wires(edge)
    for j in range(n_qubits): q.h(j)
    for gamma,beta in zip(gammas,betas):
        for u,v in edges: q.cx(u,v).rz(-gamma,v).cx(u,v)
        for j in range(n_qubits): q.rx(2*beta,j)
    return q


def _shor_modulus(modulus):
    modulus = positive_integer(modulus, 'modulus')
    if modulus < 3:
        raise ValueError('modulus must be at least 3')
    return modulus


@dataclass(frozen=True)
class ShorResources:
    """Storage estimates for noiseless order finding (not a process RAM cap)."""
    target_qubits: int
    counting_qubits: int
    total_qubits: int
    statevector_bytes: int
    estimated_working_bytes: int
    permutation_bytes: int
    estimated_construction_bytes: int


def shor_resources(modulus, counting_qubits=None, *, precision='double'):
    """Estimate resources without constructing gates or allocating a state."""
    modulus = _shor_modulus(modulus)
    if precision not in ('single', 'double'):
        raise ValueError("precision must be 'single' or 'double'")
    width = (modulus - 1).bit_length()
    n = 2 * width if counting_qubits is None else positive_integer(counting_qubits, 'counting_qubits')
    total = n + width
    if total > 62:
        raise MemoryError('Order finding exceeds 62-qubit index capacity and practical statevector storage')
    state = (8 if precision == 'single' else 16) * (1 << total)
    permutations = n * (1 << (width + 1)) * 8
    # Source maps, validation/temporary maps, prepared indices, QFT objects.
    construction = 6 * permutations + 1024 * (n * (n + 1) // 2 + n + 1)
    return ShorResources(width, n, total, state, state * 6, permutations, construction)


def order_finding(base, modulus, counting_qubits=None, *, max_memory_mb=64):
    """Build order finding using O(2**target_qubits) permutation indices per gate.

    Leading counting wires estimate s/r; trailing ceil(log2(modulus)) wires
    start in |1>. Counting width defaults to twice the target width. Unused
    basis states >= modulus are unchanged. max_memory_mb bounds estimated gate
    construction storage, not simulation storage; inspect shor_resources first.
    These logical gates are not elementary-gate hardware arithmetic circuits.
    """
    modulus = _shor_modulus(modulus)
    base = positive_integer(base, 'base')
    if not 1 < base < modulus or math.gcd(base, modulus) != 1:
        raise ValueError('base must satisfy 1 < base < modulus and be coprime to modulus')
    if isinstance(max_memory_mb, bool) or not np.isfinite(max_memory_mb) or max_memory_mb <= 0:
        raise ValueError('max_memory_mb must be positive and finite')
    resources = shor_resources(modulus, counting_qubits)
    if resources.estimated_construction_bytes > max_memory_mb * 1024**2:
        raise MemoryError('Order-finding gate construction exceeds max_memory_mb; inspect shor_resources')
    width, n = resources.target_qubits, resources.counting_qubits
    dimension = 1 << width
    q = Circuit(n + width).x(n + width - 1)
    for j in range(n):
        q.h(j)
        multiplier = pow(base, 1 << (n - 1 - j), modulus)
        mapping = np.arange(2 * dimension, dtype=np.int64)
        # Python integer arithmetic avoids multiplication overflow for large N.
        mapping[dimension:dimension + modulus] = [dimension + multiplier * x % modulus for x in range(modulus)]
        q.permutation(mapping, (j, *range(n, n + width)), 'controlled_modular_multiply')
    return q.compose(qft(n, inverse=True), range(n))


@dataclass(frozen=True)
class ShorResult:
    """A factor pair or explicit unsuccessful bounded search; factors need not be prime."""
    modulus: int
    factors: tuple | None
    attempts: int
    method: str
    base: int | None = None
    order: int | None = None


def shor(modulus, *, base=None, shots=32, max_attempts=8, seed=None, simulator=None,
         counting_qubits=None):
    """Factor an integer >=3 using resource-checked simulated order finding.

    Returns ShorResult; factors=None means the bounded attempts failed, not
    proof of primality. Even inputs and perfect powers use classical shortcuts;
    primes are recognized classically by trial division after resource checks. With
    base=None each attempt chooses a random base; a supplied base is reused.
    Each quantum attempt samples ``shots`` phases. Continued fractions and LCM
    combine their denominators; only verified modular orders produce factors.
    ``simulator`` may be a NexQus Simulator (default: NumPy statevector).
    """
    from .simulator import Simulator

    modulus = _shor_modulus(modulus)
    shots = positive_integer(shots, 'shots')
    max_attempts = positive_integer(max_attempts, 'max_attempts')
    if base is not None:
        base = positive_integer(base, 'base')
        if not 1 < base < modulus:
            raise ValueError('base must satisfy 1 < base < modulus')
    if counting_qubits is not None:
        counting_qubits = positive_integer(counting_qubits, 'counting_qubits')
    rng = np.random.default_rng(seed)
    if modulus % 2 == 0:
        return ShorResult(modulus, (2, modulus // 2), 0, 'even')
    sim = Simulator() if simulator is None else simulator
    if not isinstance(sim, Simulator):
        raise TypeError('simulator must be a Simulator')
    resources = shor_resources(modulus, counting_qubits, precision=sim.precision)
    working = resources.estimated_working_bytes
    if sim.method == 'density_matrix':
        working *= 1 << resources.total_qubits
    if working + resources.estimated_construction_bytes > sim.max_memory_mb * 1024**2:
        raise MemoryError('Shor simulation exceeds simulator.max_memory_mb; inspect shor_resources, '
                          'reduce counting_qubits, or explicitly increase the budget')
    for root in range(2, math.isqrt(modulus) + 1):
        power = root * root
        while power < modulus:
            power *= root
        if power == modulus:
            return ShorResult(modulus, (root, modulus // root), 0, 'perfect_power')
    if all(modulus % d for d in range(2, math.isqrt(modulus) + 1)):
        return ShorResult(modulus, None, 0, 'prime')
    n = resources.counting_qubits
    for attempt in range(1, max_attempts + 1):
        a = int(rng.integers(2, modulus)) if base is None else base
        divisor = math.gcd(a, modulus)
        if divisor > 1:
            return ShorResult(modulus, tuple(sorted((divisor, modulus // divisor))), attempt, 'gcd', a)
        circuit = order_finding(a, modulus, n, max_memory_mb=sim.max_memory_mb)
        result = sim.run(circuit, shots=shots, seed=int(rng.integers(0, 2**32)),
                         retain_state=False, trajectories=1)
        candidates = {1}
        for bits in result.counts:
            measurement = int(bits[:n], 2)
            if measurement == 0:
                continue
            denominator = Fraction(measurement, 1 << n).limit_denominator(modulus - 1).denominator
            # Keep alternative LCMs so one inaccurate phase cannot poison
            # later useful samples. The modulus bounds this set's size.
            candidates |= {math.lcm(c, denominator) for c in candidates
                           if math.lcm(c, denominator) < modulus}
            for candidate in sorted(candidates):
                if candidate % 2 or pow(a, candidate, modulus) != 1:
                    continue
                # Reduce a verified multiple to the actual order.
                order = candidate
                for divisor in range(2, candidate + 1):
                    while order % divisor == 0 and pow(a, order // divisor, modulus) == 1:
                        order //= divisor
                if order % 2:
                    continue
                midpoint = pow(a, order // 2, modulus)
                for divisor in (math.gcd(midpoint - 1, modulus), math.gcd(midpoint + 1, modulus)):
                    if 1 < divisor < modulus:
                        factors = tuple(sorted((divisor, modulus // divisor)))
                        return ShorResult(modulus, factors, attempt, 'quantum', a, order)
    return ShorResult(modulus, None, max_attempts, 'failed', a)
