"""Small reusable algorithms. All use the NexQus circuit engine."""
import math
import numpy as np
from .circuit import Circuit, positive_integer


def bell():
    """Prepare (|00> + |11>)/sqrt(2)."""
    return Circuit(2).h(0).cx(0,1)


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
    psi = np.array([0,1],complex) if target_state is None else np.asarray(target_state,dtype=complex)
    if psi.shape != (2,) or not np.isfinite(psi).all() or not np.isclose(np.vdot(psi,psi).real,1):
        raise ValueError("target_state must be a normalized two-element vector")
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
