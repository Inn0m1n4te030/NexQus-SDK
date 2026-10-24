"""Continuous-time transport on random-cycle glued (welded) binary trees.

The column model is an exact symmetry reduction for an entrance-localized
state, available to this classical simulator. It is not an implementation of
the paper's hidden-name coherent adjacency oracle or its query complexity.
"""
from collections import deque
import math
import numbers
import platform
import statistics
import time
import numpy as np
from .circuit import Circuit
from .maze import _integer
from .walk_device import walk_simulator, device_scope

GAMMA = 1/math.sqrt(2)  # Per-edge coupling/rate, matching the paper's Eq. 27.


def column_model(depth):
    depth = _integer(depth, 'depth', 2, 32)
    counts = np.array([2**j for j in range(depth+1)]+[2**j for j in range(depth,-1,-1)], dtype=np.int64)
    edges = np.maximum(counts[:-1], counts[1:]); edges[depth] *= 2
    couplings = GAMMA*edges/np.sqrt(counts[:-1].astype(float)*counts[1:])
    h = np.diag(couplings, 1)+np.diag(couplings, -1)
    k = np.zeros_like(h)
    for j, number in enumerate(edges):
        k[j+1,j] = GAMMA*number/counts[j]
        k[j,j+1] = GAMMA*number/counts[j+1]
    k -= np.diag(k.sum(axis=0))
    return counts, edges, h, k


def glued_graph(depth=4, seed=7):
    """Explicit bounded reference graph; leaf links form ONE alternating cycle.

    Internal indices/columns are for construction, drawing and validation only.
    bfs_hidden sees random names and neighbor lists, not indices or columns.
    """
    depth = _integer(depth, 'depth', 2, 6); seed = _integer(seed, 'seed', 0, 2**32-1)
    tree = 2**(depth+1)-1; size = 2*tree
    adjacency = [set() for _ in range(size)]; columns = []
    def edge(a,b): adjacency[a].add(b); adjacency[b].add(a)
    for offset in (0, tree):
        for parent in range(2**depth-1):
            edge(offset+parent, offset+2*parent+1); edge(offset+parent, offset+2*parent+2)
        columns.extend([(i+1).bit_length()-1 if offset==0 else 2*depth+1-((i+1).bit_length()-1)
                        for i in range(tree)])
    rng = np.random.default_rng(seed)
    left = rng.permutation(np.arange(2**depth-1, tree))
    right = rng.permutation(np.arange(tree+2**depth-1, size))
    for i in range(len(left)):
        edge(int(left[i]), int(right[i])); edge(int(right[i]), int(left[(i+1)%len(left)]))
    names = rng.choice(2**(2*depth)-1, size=size, replace=False).tolist()  # Reserve all-ones invalid name.
    neighbors = [rng.permutation(sorted(row)).tolist() for row in adjacency]
    return dict(depth=depth, seed=seed, adjacency=neighbors, columns=columns, names=names,
                entrance=0, exit=tree, vertices=size)


def bfs_hidden(graph):
    """Local classical neighbor-query BFS: given entrance name, identify other degree-2 root.

    The dictionary below emulates an oracle; building it is outside the query
    count. This is one classical strategy, not a proof about all strategies.
    """
    names = graph['names']; oracle = {names[i]: [names[j] for j in row]
                                      for i,row in enumerate(graph['adjacency'])}
    entrance = names[graph['entrance']]; queue = deque([entrance]); seen = {entrance}; queries = 0
    while queue:
        label = queue.popleft(); neighbors = oracle[label]; queries += 1
        if label != entrance and len(neighbors)==2:
            return dict(exit_name=label, neighbor_queries=queries, discovered=len(seen))
        for neighbor in neighbors:
            if neighbor not in seen: seen.add(neighbor); queue.append(neighbor)
    raise ValueError('No exit found')


class ColumnQuantumWalk:
    """Actual SDK spectral circuit for the small invariant column subspace."""
    def __init__(self, depth, *, device='cpu', backend=None):
        self.simulator = walk_simulator(device, backend)
        self.counts, self.edges, self.h, self.k = column_model(depth)
        self.depth = depth; self.dimension = len(self.counts)
        self.n_qubits = (self.dimension-1).bit_length(); size = 2**self.n_qubits
        self.values, self.vectors = np.linalg.eigh(self.h)
        basis = np.eye(size, dtype=complex); basis[:self.dimension,:self.dimension] = self.vectors
        self.energies = np.zeros(size); self.energies[:self.dimension] = self.values
        self.to_energy = Circuit(self.n_qubits).unitary(basis.conj().T, range(self.n_qubits), 'column_to_energy')
        self.to_columns = Circuit(self.n_qubits).unitary(basis, range(self.n_qubits), 'energy_to_column')

    def circuit(self, evolution_time):
        if isinstance(evolution_time, bool) or not isinstance(evolution_time, numbers.Real):
            raise ValueError('time must be a finite nonnegative real number')
        evolution_time = float(evolution_time)
        if not math.isfinite(evolution_time) or not 0 <= evolution_time <= 1e6:
            raise ValueError('time must be in 0..1000000')
        q = Circuit(self.n_qubits)
        q.operations.extend(self.to_energy.operations)
        q.diagonal(np.exp(-1j*self.energies*evolution_time), range(self.n_qubits), 'evolve_energy_phase')
        q.operations.extend(self.to_columns.operations)
        return q

    def states(self, times):
        simulator = self.simulator
        return np.array([simulator.run(self.circuit(float(t)), shots=0).statevector[:self.dimension] for t in times])

    def uniform_time_exit_probability(self, horizon):
        # Exact integral of |sum_k c_k exp(-i E_k t)|² over t in [0,T].
        c = self.vectors[-1]*self.vectors[0]
        delta = self.values[:,None]-self.values[None,:]
        return float(np.sum(c[:,None]*c[None,:]*np.sinc(delta*horizon/np.pi)))


def classical_transport(k, times, horizon):
    """Positive uniformization of exp(Kt); no cancellation of tiny exit mass.

    K is the per-edge-rate CTMC on columns, not a symmetric amplitude matrix.
    Poisson expansion has a conservative 12-sigma + 32 term tail cutoff.
    Uniform-time weights are computed with reverse tails, avoiding 1-CDF loss.
    """
    rate = float(max(-np.diag(k))); transition = np.eye(len(k))+k/rate
    means = rate*np.asarray(times); maximum = rate*horizon
    terms = math.ceil(maximum+12*math.sqrt(maximum+1)+32)
    weights = np.empty(terms+2); weights[0] = math.exp(-maximum)
    for j in range(1,len(weights)): weights[j] = weights[j-1]*maximum/j
    average_weights = np.cumsum(weights[::-1])[::-1][1:]/maximum
    state = np.zeros(len(k)); state[0] = 1
    poisson = np.exp(-means); distributions = np.zeros((len(times),len(k))); average = np.zeros(len(k))
    for j in range(terms+1):
        distributions += poisson[:,None]*state[None,:]; average += average_weights[j]*state
        state = transition@state; poisson *= means/(j+1)
    return distributions, average


def classical_mean_hitting_time(counts, edges):
    # Reversible birth/death-chain first-passage formula; rate per edge GAMMA.
    return float(np.sum(np.cumsum(counts)[:-1]/(GAMMA*edges)))


def verify_full_graph(depth=3, seed=7):
    """Independent full-vertex spectral evolution versus actual SDK column states."""
    graph = glued_graph(depth,seed); m = 2*depth+2; size = graph['vertices']
    a = np.zeros((size,size))
    for i,row in enumerate(graph['adjacency']): a[row,i] = 1
    counts,edges,h,k = column_model(depth)
    basis = np.zeros((size,m))
    for i,j in enumerate(graph['columns']): basis[i,j] = 1/math.sqrt(int(counts[j]))
    invariance_error = float(np.max(abs(GAMMA*a@basis-basis@h)))
    values,vectors = np.linalg.eigh(GAMMA*a)
    times = np.linspace(0, 2*(depth+1), 5)
    full = (np.exp(-1j*times[:,None]*values[None,:])*vectors[graph['entrance']])@vectors.T
    reduced = ColumnQuantumWalk(depth).states(times)@basis.T
    return dict(depth=depth,seed=seed,vertices=size,invariance_max_error=invariance_error,
                state_max_error=float(np.max(abs(full-reduced))),
                bfs=bfs_hidden(graph))


def experiment(depth=4, samples=129, *, device='cpu', backend=None):
    depth = _integer(depth,'depth',2,32); samples = _integer(samples,'samples',17,257)
    # Set horizon before observing data; no peak-time optimization in primary metric.
    horizon = 4.*(depth+1); times = np.linspace(0,horizon,samples)
    model = ColumnQuantumWalk(depth, device=device, backend=backend); states = model.states(times)
    classical, average_classical = classical_transport(model.k,times,horizon)
    average_quantum = model.uniform_time_exit_probability(horizon)
    peak = int(np.argmax(abs(states[:,-1])**2))
    timings = {key:[] for key in ('quantum_ms','classical_ms')}
    for trial in range(7):
        model.simulator.backend.sync()
        start = time.perf_counter_ns()
        prepared = ColumnQuantumWalk(depth, device=device, backend=backend)
        prepared.uniform_time_exit_probability(horizon)
        prepared.states([horizon/2])
        prepared.simulator.backend.sync()
        qend = time.perf_counter_ns()
        _,_,_,generator = column_model(depth)
        classical_transport(generator,[horizon/2],horizon)
        end = time.perf_counter_ns()
        if trial>=2:
            timings['quantum_ms'].append((qend-start)/1e6)
            timings['classical_ms'].append((end-qend)/1e6)
    return dict(depth=depth, vertices=2**(depth+2)-2, columns=len(model.counts),
        column_sizes=model.counts.tolist(), column_qubits=model.n_qubits,
        compact_vertex_qubits=depth+2, hidden_name_bits=2*depth, gamma=GAMMA,
        horizon=horizon, horizon_rule='T=4*(depth+1), fixed before evaluation', times=times.tolist(),
        quantum_probabilities=(abs(states)**2).tolist(), real=states.real.tolist(), imag=states.imag.tolist(),
        classical_probabilities=classical.tolist(), uniform_quantum_exit=average_quantum,
        uniform_classical_exit=float(average_classical[-1]),
        classical_mean_first_hit=classical_mean_hitting_time(model.counts,model.edges),
        quantum_ideal_repeat_evolution_time=horizon/(2*average_quantum),
        sampled_peak=dict(time=float(times[peak]),probability=float(abs(states[peak,-1])**2),
                          scope='Descriptive grid maximum, not the measurement-time selection rule'),
        circuit_svg=str(model.circuit(horizon/2).draw('svg')),
        timing=dict(median_ms={key:statistics.median(v) for key,v in timings.items()},samples_ms=timings,
            warmups=2,repeats=5,python=platform.python_version(),numpy=np.__version__,platform=platform.platform(),
            device=model.simulator.backend.device, backend=model.simulator.backend.name,
            scope=device_scope(model.simulator)+'Wall time: setup, uniform-time success calculation and one state at T/2. '
                  'Quantum includes eigendecomposition, SDK circuit construction/validation and statevector execution. '
                  'Classical includes generator construction and positive Poisson uniformization. '
                  'No graph generation, full-graph verification, plots or export. '
                  'Both models use known column symmetry. Not oracle-query cost or QPU time.'))


def compare_depths(depths=(2,4,8,16,32), seed=7, samples=129, *, device='cpu', backend=None):
    if not isinstance(depths,(list,tuple)) or not 1<=len(depths)<=8:
        raise ValueError('Supply 1..8 distinct depths')
    depths = [_integer(d,'depth',2,32) for d in depths]
    if len(set(depths))!=len(depths): raise ValueError('Depths must be distinct')
    seed = _integer(seed,'seed',0,2**32-1)
    results = [experiment(depth,samples,device=device,backend=backend) for depth in depths]
    verification = [verify_full_graph(depth,seed) for depth in (2,3,4)]
    # Bounded full graph is for visual explanation and classical oracle queries.
    graph = glued_graph(4,seed)
    return dict(results=results,seed=seed,graph=graph,verification=verification,
                graph_bfs=bfs_hidden(graph),reference='https://arxiv.org/abs/quant-ph/0209131',
                protocol='Reset to entrance; choose t uniformly in [0,T]; evolve unmeasured; measure once. '
                         'Success is observing the exit vertex. Curves show independent final-time experiments, '
                         'not cumulative detection. The primary metric analytically averages over time. '
                         'No absorbing exit and no continuous monitoring. No route extraction.')
