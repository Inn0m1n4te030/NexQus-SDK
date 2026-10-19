# Quantum Maze Explorer — a local school project

Draw a maze, compare classical BFS with a small Grover experiment, and inspect
the actual circuit states. It runs on NumPy CPU with no GPU, paid QPU, cloud
account, web framework or internet connection after installation.

This is an **unreleased source-checkout example**, not an API in the existing
PyPI 1.2.0 wheel. From the checkout containing this change:

```bash
python -m pip install -e .
python examples/quantum_maze.py
```

The command opens a local browser page at `http://127.0.0.1:8765`.
Keep the terminal open; Ctrl+C stops the server. If that port is busy, run
`python examples/quantum_maze.py --port 0` and use the printed address.
`--no-browser` prints the address without opening a tab. Native Windows,
macOS and Linux use the same commands with Python 3.10+.

## Try the lesson

1. Start with **Two corridors**. Use Wall / erase, Entrance S or Exit G to
   edit cells. Every edit invalidates the old result; click **Run NexQus**.
2. Inspect the 64 candidate plans. Two bits per junction give six qubits for
   a three-junction budget. Choose a plan to draw its path and inspect its decisions.
3. **Prepare / reset** shows equal probabilities. **Next stage** first applies
   the phase oracle: valid amplitudes change sign, while probabilities stay equal.
   The following stage applies diffusion, making interference change the odds.
4. Play or step through the iterations. In the default ideal example, two of
   64 encoded plans are valid. Total valid-plan probability grows from 3.125%
   after preparation to about 99.918% after four Grover iterations.
5. **Measure this stage** uses the SDK's seeded sampling on that stage's
   actual circuit prefix. It can return an invalid plan, especially before
   amplification. Live measurements increment the seed; set it back to repeat.
   Each shot prepares that stage again; the plots remain its pre-measurement
   state rather than displaying a collapsed state.
6. Overlay the BFS shortest path. Then try eight iterations to see overshoot,
   Blocked exit to see a genuinely unreachable exit, and Choice-budget challenge
   to see a reachable maze that the bounded encoding cannot solve.
7. Download the experiment JSON or the standalone gate-by-gate circuit lesson.
   The circuit SVG comes from `Circuit.draw`; the lesson comes from `Circuit.explain`.

Probability bars have a fixed 0–100% scale. Signed real-amplitude plots label
their changing vertical range. These circuits contain real-valued operations,
so a negative amplitude represents relative phase π. It is not a negative
probability. All plots are computed from NexQus simulator output.

## How plans are encoded

At each cell, the classical checker lists **unvisited**, open neighbors in
right/down/left/up order. With one option it follows the corridor automatically.
With multiple options, the next pair of bits selects option 0, 1, 2 or 3:

| Bits | Selection from the available list |
|---|---|
| 00 | First option |
| 01 | Second option |
| 10 | Third option |
| 11 | Fourth option |

These pairs are **option indices**, not fixed compass directions. Selecting a
missing option, reaching a dead end, or needing too many junction choices rejects
the plan. Cells are not revisited. Once G is reached, unused bit pairs must be
00, so the same route is not counted under multiple padded encodings.

Bounds are 2–8 rows and columns, one S and one G, 1–3 junction choices
(2–6 qubits, 4–64 plans), and 0–8 Grover iterations. More elaborate mazes may
need more junction decisions than this school-project encoding supports.

## What is quantum, and what is classical?

Python checks **every encoded plan** and builds an explicit phase-oracle truth
table. It does not use BFS or a precomputed shortest path to mark the oracle.
NexQus then simulates Hadamard preparation, the phase oracle and diffusion.
The UI displays snapshots from actual circuits, and stage measurement calls the
SDK's simulator with one shot. The oracle is a logical diagonal operation,
not a reversible gate-level implementation of maze checking.

The report separately counts BFS node expansions, classical plan checks and
route-cell visits, and logical quantum oracle calls. These are different units.
Logical calls refer to the final circuit. Tracing and example samples rerun
circuit prefixes, so total simulator work exceeds that logical count.
The classical compiler has already inspected the whole encoded search space;
the project therefore does **not** demonstrate an end-to-end speedup over BFS.
No runtime chart or quantum-advantage verdict is inferred.

BFS is complete on the grid and returns a shortest path. Grover marks valid
encoded routes without minimizing their length. A valid sampled route may be
longer than BFS's route. Zero valid encodings can mean either an unreachable
exit or insufficient choice budget; the UI distinguishes those cases.

## Share a self-contained offline lesson

```bash
python examples/quantum_maze.py --export maze_lesson.html
```

Open that file directly in a browser, even without Python on the viewing computer.
It includes states, plots, the circuit and the downloadable circuit lesson.
Offline editing/rerunning is disabled: custom mazes need the Python server.
Offline measurement reveals one precomputed SDK sample per stage at the chosen
seed, rather than silently replacing the SDK with a JavaScript simulator.
Choose another reproducible export with `--seed 11`, `--junctions 2`,
`--iterations 3`, or `--preset "Long hallway"`.

## Use the Python experiment directly

```python
from nexqus.maze import Maze, PRESETS, explore, breadth_first

maze = Maze(PRESETS['Two corridors'])
experiment = explore(maze, junctions=3)
print('BFS shortest path:', breadth_first(maze)['path'])
print('Valid-plan probability:', experiment.frames[-1]['success_probability'])
print('One seeded sample:', experiment.measure(seed=7))
experiment.circuit.draw('svg', filename='maze_circuit.svg')
```

`experiment.frames` records every prepared/oracle/diffusion stage, signed
amplitudes, probabilities and operation stops. `experiment.measure(frame=0,
seed=7)` samples the prepared stage. `experiment.to_dict()` includes all candidate
paths, validation reasons, resource counts and the separate BFS result.
`experiment.explanation()` returns the SDK's standalone circuit explanation.

## Verification

```bash
python -m unittest discover -s tests -p test_maze.py -v
python examples/quantum_maze.py --export .release-artifacts/maze/lesson.html
npm install --no-save --no-package-lock jsdom@26.1.0
node scripts/check_maze_ui.cjs
```

NumPy tests independently check phase flips, diffusion about the mean, the
analytic Grover probability formula, parity with the SDK's existing Grover
algorithm, seeded unfiltered measurement, maze paths and bounds. The UI test
also starts a real Python server and exercises editing, oracle rebuilding,
navigation, playback, stage measurement, downloads and no-path/budget cases.
jsdom is a development-only UI test dependency; students do not need Node.
