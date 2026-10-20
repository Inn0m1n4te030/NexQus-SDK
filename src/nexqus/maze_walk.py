"""A monitored coined walk on maze cells, and its fully dephased baseline.

This is an exit-detection experiment, not a shortest-path extraction algorithm.
No route enumeration or solution-dependent oracle is used. The entire grid is
known to both methods. Simulation and detector bookkeeping run locally on CPU.
"""
import platform
import statistics
import time
import numpy as np
from .circuit import Circuit
from .simulator import Simulator
from .maze import Maze, PRESETS, _DIRECTIONS, _integer, breadth_first


WALK_PRESETS = dict(PRESETS, **{
    'Open 6 by 6': ('S.....', '......', '......', '......', '......', '.....G'),
    'Snake 8 by 8': ('S.......', '#######.', '........', '.#######',
                     '........', '#######.', '........', 'G#######'),
})


def walk_operators(maze):
    """Position in high bits; R,D,L,U ports in the low two bits.

    C=2|s><s|-I on four ports. S swaps opposite ports across open edges;
    blocked ports stay fixed (reflecting self-loops). Thus S is an involution.
    Wall/padded position states never receive amplitude from accessible cells.
    """
    if not isinstance(maze, Maze): raise TypeError('Expected a Maze')
    width = len(maze.rows[0]); cells = len(maze.rows)*width
    position_qubits = (cells-1).bit_length(); size = 4*(2**position_qubits)
    mapping = np.arange(size, dtype=np.int64)
    for r, row in enumerate(maze.rows):
        for c, symbol in enumerate(row):
            if symbol == '#': continue
            for direction, (dr, dc) in enumerate(_DIRECTIONS):
                neighbor = (r+dr, c+dc)
                if neighbor in maze.neighbors((r,c)):
                    mapping[4*(r*width+c)+direction] = 4*(neighbor[0]*width+neighbor[1])+(direction+2)%4
    coin = np.full((4,4), .5)-np.eye(4)
    circuit = Circuit(position_qubits+2)
    circuit.unitary(coin, (position_qubits, position_qubits+1), 'Grover_coin')
    circuit.permutation(mapping, range(position_qubits+2), 'flip_flop_shift')
    initial = np.zeros(size, complex)
    start = 4*(maze.start[0]*width+maze.start[1]); initial[start:start+4] = .5
    goal = maze.goal[0]*width+maze.goal[1]
    return circuit, initial, mapping, goal


def _quantum(circuit, initial, goal, steps, trace=False):
    simulator = Simulator(backend='numpy', method='statevector')
    survivor = initial.copy(); detected = 0.; frames = []
    if trace: frames.append(dict(real=survivor.real.tolist(), imag=survivor.imag.tolist(),
        probabilities=(abs(survivor.reshape(-1,4))**2).sum(axis=1).tolist(),
        hit=0., cumulative_hit=0.))
    for _ in range(steps):
        mass = float(np.vdot(survivor, survivor).real)
        if mass > 1e-28:
            # SDK input is normalized conditional on no earlier detection.
            # Restore unconditional weight before the next detector projection.
            survivor = simulator.run(circuit, initial_state=survivor/np.sqrt(mass), shots=0).statevector*np.sqrt(mass)
        hit = float(np.sum(abs(survivor[4*goal:4*goal+4])**2))
        detected += hit; survivor[4*goal:4*goal+4] = 0
        if trace: frames.append(dict(real=survivor.real.tolist(), imag=survivor.imag.tolist(),
            probabilities=(abs(survivor.reshape(-1,4))**2).sum(axis=1).tolist(),
            hit=hit, cumulative_hit=detected))
    return frames, detected


def _classical(initial, mapping, goal, steps, trace=False):
    # Fully dephase the same Grover coin: every |C_ij|^2 = 1/4.
    survivor = abs(initial)**2; detected = 0.; frames = []
    if trace: frames.append(dict(probabilities=survivor.reshape(-1,4).sum(axis=1).tolist(),
                                 hit=0., cumulative_hit=0.))
    for _ in range(steps):
        mixed = np.repeat(survivor.reshape(-1,4).sum(axis=1)/4, 4)
        survivor = np.empty_like(mixed); survivor[mapping] = mixed
        hit = float(survivor[4*goal:4*goal+4].sum())
        detected += hit; survivor[4*goal:4*goal+4] = 0
        if trace: frames.append(dict(probabilities=survivor.reshape(-1,4).sum(axis=1).tolist(),
                                     hit=hit, cumulative_hit=detected))
    return frames, detected


def compare_walks(maze, steps=32):
    """Same start, grid, reflecting walls and after-every-step exit detector.

    Exact distributions, no Monte Carlo or best-time selection. Nine full-work
    CPU trials after two warmups, including each method's grid preparation.
    BFS returns a full shortest path; the walkers only detect arrival at G.
    """
    steps = _integer(steps, 'steps', 1, 128)
    circuit, initial, mapping, goal = walk_operators(maze)
    qframes, qhit = _quantum(circuit, initial, goal, steps, trace=True)
    cframes, chit = _classical(initial, mapping, goal, steps, trace=True)
    times = {key: [] for key in ('bfs_ms', 'quantum_ms', 'classical_walk_ms')}
    for trial in range(11):
        start = time.perf_counter_ns(); breadth_first(maze); bfs_end = time.perf_counter_ns()
        q, state, shift, target = walk_operators(maze)
        _quantum(q, state, target, steps); qend = time.perf_counter_ns()
        # Use the same grid builder for matched preparation cost, including its
        # coin validation; this is a conservative classical implementation.
        _, state, shift, target = walk_operators(maze)
        _classical(state, shift, target, steps); end = time.perf_counter_ns()
        if trial >= 2:
            for key, duration in zip(times, (bfs_end-start, qend-bfs_end, end-qend)):
                times[key].append(duration/1e6)
    classical = breadth_first(maze)
    return dict(rows=list(maze.rows), steps=steps, qubits=circuit.n_qubits,
        quantum=qframes, classical_walk=cframes, bfs=classical,
        quantum_hit=qhit, classical_hit=chit,
        difference=qhit-chit,
        quantum_50_step=next((i for i,f in enumerate(qframes) if f['cumulative_hit']>=.5), None),
        classical_50_step=next((i for i,f in enumerate(cframes) if f['cumulative_hit']>=.5), None),
        logical_operations_per_step=2, shift_is_logical=True,
        circuit_svg=str(circuit.draw('svg')),
        timings=dict(median_ms={key:statistics.median(v) for key,v in times.items()},
            samples_ms=times, warmups=2, repeats=9, python=platform.python_version(),
            numpy=np.__version__, platform=platform.platform(),
            scope=f'CPU wall time, {steps} monitored walk steps or one complete BFS. '
                  'Each walk includes grid/shift construction and initial-state preparation. '
                  'Classical uses the same builder, including coin validation. '
                  'Quantum includes SDK simulation and detector projection. '
                  'No frame capture, plots, export or network. Exact probabilities, no sampled shots. '
                  'Walks detect the exit; BFS returns a shortest route. No QPU execution.'))
