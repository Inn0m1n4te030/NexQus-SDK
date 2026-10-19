"""Bounded school-project maze experiments using actual NexQus circuits.

Two qubits encode each junction choice; ordinary code follows corridors and
constructs an explicit oracle truth table. This is an amplitude-amplification
lesson, not an end-to-end maze-solving speedup or a gate-decomposed path oracle.
"""
from collections import deque
from dataclasses import dataclass
import math
import numbers
import numpy as np
from .circuit import Circuit
from .simulator import Simulator


PRESETS = {
    'Two corridors': ('S...', '.##.', '....', '.##G'),
    'Long hallway': ('S....', '####.', '.....', '.####', '....G'),
    'Blocked exit': ('S...', '.###', '.#G#', '.###'),
    'Choice-budget challenge': ('S...', '....', '....', '...G'),
}
_DIRECTIONS = ((0, 1), (1, 0), (0, -1), (-1, 0))  # Right, down, left, up.


def _integer(value, name, minimum, maximum):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or not minimum <= value <= maximum:
        raise ValueError(f'{name} must be an integer in {minimum}..{maximum}')
    return int(value)


@dataclass(frozen=True)
class Maze:
    rows: tuple

    def __post_init__(self):
        if not isinstance(self.rows, (tuple, list)) or not 2 <= len(self.rows) <= 8:
            raise ValueError('Supply 2..8 maze rows')
        rows = tuple(self.rows)
        if (any(not isinstance(row, str) for row in rows) or not 2 <= len(rows[0]) <= 8
                or any(len(row) != len(rows[0]) or set(row)-set('.#SG') for row in rows)):
            raise ValueError('Use equal-width rows of 2..8 characters: . # S G')
        if ''.join(rows).count('S') != 1 or ''.join(rows).count('G') != 1:
            raise ValueError('The maze needs exactly one S entrance and one G exit')
        object.__setattr__(self, 'rows', rows)

    def locate(self, symbol):
        return next((r, c) for r, row in enumerate(self.rows) for c, value in enumerate(row) if value == symbol)

    @property
    def start(self): return self.locate('S')

    @property
    def goal(self): return self.locate('G')

    def neighbors(self, cell):
        r, c = cell
        return [(r+dr, c+dc) for dr, dc in _DIRECTIONS
                if 0 <= r+dr < len(self.rows) and 0 <= c+dc < len(self.rows[0])
                and self.rows[r+dr][c+dc] != '#']


def breadth_first(maze):
    """Complete four-direction BFS, returning a shortest path and node counts."""
    queue = deque([maze.start]); parents = {maze.start: None}; expanded = 0
    while queue:
        cell = queue.popleft(); expanded += 1
        if cell == maze.goal:
            path = []
            while cell is not None: path.append(cell); cell = parents[cell]
            return dict(path=path[::-1], expanded_nodes=expanded, discovered_nodes=len(parents))
        for neighbor in maze.neighbors(cell):
            if neighbor not in parents: parents[neighbor] = cell; queue.append(neighbor)
    return dict(path=[], expanded_nodes=expanded, discovered_nodes=len(parents))


def check_plan(maze, bits, junctions=3):
    """Decode without BFS or any precomputed solution.

    Available unvisited neighbors are ordered right/down/left/up. Corridors
    require no bits. At a junction, 00/01/10/11 select the available neighbor
    with that index. Unused groups must be 00, giving each completed route a
    single canonical encoding. Exhausted choices do not prove no maze path exists.
    """
    junctions = _integer(junctions, 'junctions', 1, 3)
    if not isinstance(bits, str) or len(bits) != 2*junctions or set(bits)-set('01'):
        raise ValueError('Supply exactly two binary bits per junction')
    cell = maze.start; path = [cell]; visited = {cell}; used = 0; decisions = []
    while cell != maze.goal:
        options = [p for p in maze.neighbors(cell) if p not in visited]
        if not options: reason = 'Dead end (visited cells are not revisited)'; break
        pick = 0
        if len(options) > 1:
            if used == junctions: reason = 'Junction-choice budget exhausted'; break
            pair = bits[2*used:2*used+2]; pick = int(pair, 2); used += 1
            decisions.append(dict(cell=cell, options=options, bits=pair, selected=pick))
            if pick >= len(options): reason = 'Choice index is unavailable at this junction'; break
        cell = options[pick]; visited.add(cell); path.append(cell)
    else:
        reason = 'Valid route' if '1' not in bits[2*used:] else 'Exit reached, but unused choices must be 00'
    valid = reason == 'Valid route'
    return dict(bits=bits, index=int(bits, 2), path=path, valid=valid, reached_goal=cell == maze.goal,
                reason=reason, choices_used=used, decisions=decisions, moves=len(path)-1)


@dataclass
class MazeExperiment:
    maze: Maze
    junctions: int
    iterations: int
    suggested_iterations: int
    candidates: list
    frames: list
    circuit: Circuit

    def measure(self, frame=None, seed=7):
        """Sample the displayed stage using NexQus, including an unmarked outcome."""
        seed = _integer(seed, 'seed', 0, 2**32-1)
        frame = len(self.frames)-1 if frame is None else _integer(frame, 'frame', 0, len(self.frames)-1)
        prefix = Circuit(self.circuit.n_qubits)
        prefix.operations.extend(self.circuit.operations[:self.frames[frame]['operation_stop']])
        sampled = Simulator(backend='numpy', method='statevector').run(prefix, shots=1, seed=seed)
        bits = next(iter(sampled.counts)); candidate = self.candidates[int(bits, 2)]
        return dict(bits=bits, index=int(bits, 2), valid=candidate['valid'], path=candidate['path'],
                    reason=candidate['reason'], frame=frame, seed=seed)

    def to_dict(self, seed=7, offline=False):
        valid = sum(p['valid'] for p in self.candidates); classical = breadth_first(self.maze)
        return dict(rows=self.maze.rows, junctions=self.junctions, qubits=self.circuit.n_qubits,
            iterations=self.iterations, suggested_iterations=self.suggested_iterations,
            candidates=self.candidates, frames=self.frames, valid_plans=valid,
            classical=classical, offline=offline, seed=seed,
            resources=dict(classical_route_checks=len(self.candidates),
                classical_route_cell_visits=sum(len(p['path']) for p in self.candidates),
                logical_oracle_calls=self.iterations, logical_depth=self.circuit.depth(),
                logical_gate_counts=self.circuit.count_ops()),
            circuit_svg=str(self.circuit.draw('svg', fold=14)),
            samples=[self.measure(i, seed) for i in range(len(self.frames))],
            presets={name: list(rows) for name, rows in PRESETS.items()})

    def explanation(self):
        return str(self.circuit.explain(title='Quantum Maze Explorer — inside the actual circuit',
            max_qubits=6, max_steps=128, backend='numpy', device='cpu').to_html())


def explore(maze, junctions=3, iterations=None):
    """Prepare an explicit phase oracle and calculate every displayed SDK stage.

    At most 64 candidate plans and eight Grover iterations. The classical oracle
    compiler already checks every plan; its cost is disclosed, not hidden in a
    query-complexity claim. BFS is a separate baseline and never marks the oracle.
    """
    if not isinstance(maze, Maze): raise TypeError('Expected a Maze')
    junctions = _integer(junctions, 'junctions', 1, 3); n = 2*junctions; size = 2**n
    candidates = [check_plan(maze, format(index, f'0{n}b'), junctions) for index in range(size)]
    marked = np.array([candidate['valid'] for candidate in candidates], dtype=bool)
    count = int(marked.sum()); suggested = 0
    if 0 < count < size:
        theta = math.asin(math.sqrt(count/size)); suggested = max(0, round(math.pi/(4*theta)-.5))
    iterations = suggested if iterations is None else _integer(iterations, 'iterations', 0, 8)
    q = Circuit(n); frames = []; simulator = Simulator(backend='numpy', method='statevector')
    def snapshot(phase, iteration):
        result = simulator.run(q, shots=0)
        frames.append(dict(phase=phase, iteration=iteration, operation_stop=len(q.operations),
            real=result.statevector.real.tolist(), imag=result.statevector.imag.tolist(),
            probabilities=result.probabilities.tolist(), success_probability=float(result.probabilities[marked].sum())))
    for wire in range(n): q.h(wire)
    snapshot('prepare', 0)
    oracle = np.where(marked, -1., 1.)
    reflection = -np.ones(size); reflection[0] = 1.
    for iteration in range(1, iterations+1):
        q.diagonal(oracle, range(n), 'maze_oracle'); snapshot('mark', iteration)
        for wire in range(n): q.h(wire)
        q.diagonal(reflection, range(n), 'diffusion_phase')
        for wire in range(n): q.h(wire)
        snapshot('amplify', iteration)
    return MazeExperiment(maze, junctions, iterations, suggested, candidates, frames, q)
