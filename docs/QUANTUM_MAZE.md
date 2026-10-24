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
The lesson measures local CPU runtime without inferring quantum advantage.

### Step through both algorithms

The classical algorithm is **breadth-first search (BFS)**. Its FIFO queue starts
with S. Each visit removes the oldest cell, checks its open neighbors, and adds
unseen neighbors to the back. Parent pointers reconstruct the route when G is
removed. Use Previous/Next visit or the BFS slider to inspect the actual queue,
newly discovered cells and expanded cells. Enable the exploration overlay to see
expanded cells in cyan and waiting cells in purple. Coordinates are zero-based.

The quantum algorithm is **Grover amplitude amplification**, a gate-based
search over encoded route instructions. It is neither quantum annealing nor a
quantum walk. Hadamards prepare equal amplitudes; the oracle changes valid-plan
amplitude signs without changing their probabilities; diffusion reflects every
amplitude around the mean. Interference concentrates probability on marked
plans until the optimum iteration count. One measurement returns one plan,
not all the plans or a guaranteed answer. The selected-plan arithmetic updates
with each stage and candidate selection.

For the default maze:

| Quantity | Actual count |
| --- | ---: |
| BFS queue removals (including S and G) | 12 |
| BFS open-neighbor checks | 23 |
| Shortest route moves | 6 |
| Encoded plans checked classically for the oracle | 64 |
| Classical compiler route-cell visits | 215 |
| Qubits | 6 |
| Grover iterations / oracle calls | 4 |
| Logical operations | 62 (54 H, 4 oracle, 4 diffusion phase) |
| Logical depth | 17 |
| Valid-plan probability before / after | 3.125% / approximately 99.918% |

These units are not interchangeable. The logical diagonal operations do not
represent a hardware-native decomposition. A maze move is different from a BFS
visit or a Grover iteration. At the first marking stage, a valid plan's amplitude
changes from +0.125 to −0.125. The mean becomes 0.1171875; diffusion produces
2 × 0.1171875 − (−0.125) = 0.359375, making that plan's probability about 12.915%.
Both marked plans together then have approximately 25.830% probability.

### What the time comparison measures

Each solve/export reports medians of nine runs after two untimed warmups of
**both** methods (including the sampler). BFS time includes solving and path
reconstruction. Quantum total includes enumerating/checking all plans, building
the oracle and circuit, one complete CPU statevector simulation, one seeded
shot and its validity lookup. Compilation and simulation medians also appear
separately; their sum need not equal the median total. Raw samples, environment,
precision and timing scope are included in the JSON download.

Tracing, SVG/HTML export, network and animation are excluded. The benchmark uses
the same circuit builder as the lesson but disables educational prefix replays.
The result is **local NumPy CPU time**, not QPU time or a hardware-speedup
prediction. A tiny maze is well suited to BFS. A quantum shot can fail and need
repetition; the table reports one attempt, not time to guaranteed success.
Timings vary by machine and load. Offline lessons retain their export-time
measurements; running the Python example measures on your own machine.

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
paths, validation reasons, resource counts, warmed timing samples and the
separate BFS result with its queue trace. Use `breadth_first(maze, trace=True)`
to request BFS frames directly; the default omits them.
`experiment.explanation()` returns the SDK's standalone circuit explanation.

## Try a quantum walk instead of route-plan search

```bash
python examples/quantum_maze_walk.py --steps 32 --open
python examples/quantum_maze_walk.py --steps 128 --open
python examples/quantum_maze_walk.py --preset "Open 6 by 6" --steps 64 --open
```

The command writes `.release-artifacts/maze-walk/walk.html` and `walk.json`.
The report starts with a seeded run animation: a classical walker and its trail
beside a quantum probability wave, outgoing coin-flow arrows and an exit
detector. Use **Run until exit**, **One step**, speed selection and reset.
The default horizon is now 128 steps. Each walker stops at its sampled exit
event; playback ends when both finish or the horizon is exhausted.
The quantum run samples first-detection time from the SDK's cumulative exit
distribution. Before detection, circles show the normalized no-detection branch;
at detection the display collapses onto G. It does not draw an invented quantum
trajectory. The classical dot samples uniform directional moves, including wall
self-loops. The existing distribution charts remain below for inspection.
The HTML is self-contained. `--open` opens it in your browser; without that flag
the command prints its path. Change `--output` to retain separate experiments.
For custom grids, save a plain text file with `. # S G` rows and pass
`--maze my_maze.txt`. Bounds remain 2..8 rows/columns, with 1..128 walk steps.
No junction-choice budget applies to this experiment.

This uses a **discrete-time four-port coined quantum walk** on cell positions,
not the earlier global Grover search. The basis is `|position, direction>`;
directions are R/D/L/U in the two low bits. A local Grover coin
`C = 2|s><s| - I` mixes the four port amplitudes. A flip-flop shift moves an
open port into its neighbor's opposite port. Blocked ports are reflecting
self-loops. Padded positions and wall cells are unreachable from S.
The coin is an actual two-qubit SDK unitary, and the shift uses the SDK's
permutation operation. There is no route enumeration or solution oracle.
Shift construction still reads the entire known grid; it is not a native-gate
decomposition or a black-box graph-access implementation.

After every step, a two-outcome detector asks whether the walker is at G.
Probability detected there is accumulated and removed. The surviving branch
keeps its coherence. Internally each surviving SDK state is normalized for
execution, then restored to its unconditional weight. The displayed cell masses
are **not** renormalized to 100%: remaining mass plus cumulative detection is
100%. Measuring all cell positions each step would destroy the interference.

The matched classical walk is the **fully dephased version of the same coin
and shift**. Every squared coin entry is 1/4, so each step chooses a port
uniformly, stays put at blocked ports, and applies the same exit detector.
Both initial conditions are uniform over S's ports. Classical probabilities
are propagated exactly, not estimated from a noisy sample of random walkers.
This classical walk is not intended as the strongest classical maze solver;
BFS is shown separately and returns a full shortest route.

### Recorded experiments

The following use a fixed 32-step horizon chosen before comparison; all six
included presets are reported. Values are exact CPU-simulated probabilities
rounded for display, not QPU measurements.

| Maze | Quantum exit detection by step 32 | Classical walk detection | BFS route moves |
| --- | ---: | ---: | ---: |
| Two corridors (4×4) | 60.713% | 19.473% | 6 |
| Long hallway (5×5) | 86.718% | 0.006656% | 16 |
| Blocked exit (4×4) | 0% | 0% | No route |
| Choice-budget challenge (open 4×4) | 86.216% | 36.713% | 6 |
| Open 6×6 | 60.638% | 7.109% | 10 |
| Snake 8×8 | 0% | 0% | 35 |

The snake cannot reach G within 32 local shifts: its shortest route takes 35
moves. For Two corridors, a separate 128-step run reaches 50% cumulative
detection at step **18 quantum versus 74 classical**. At step 128 the respective
probabilities are 90.086% and 73.470%. These are **finite-instance benefits over
this random-walk baseline**, not evidence of a scaling advantage or a BFS
speedup. Different coins, boundary conditions, initialization, monitoring
schedules and graphs may change the outcome; interference can trap amplitude.

Both walk times include initialization, grid/shift construction and all steps.
For conservative matched preparation, the classical implementation uses the
same builder including coin validation. Each has two warmups and nine measured
trials. Times exclude trace capture and chart export. The quantum SDK CPU
simulation was slower than BFS and the classical probability propagation in
these trials. The report lists all three times and exports raw timing samples.

**Detecting the exit does not recover a route.** Measuring position provides
the endpoint, not the sequence traversed. No shortest-path extraction is
implemented for the quantum walk. The report shows survivor heatmaps, cumulative
detection curves for every step, and clickable local coin arithmetic.

Relevant research: [Childs et al., STOC 2003](https://arxiv.org/abs/quant-ph/0209131)
proves exponential speedup for a specially constructed black-box graph problem.
That access model and graph are different from our known grid. The maze
framework of [Matsuoka, Ohno and Segawa](https://arxiv.org/abs/2411.12191) uses
absorbing holes and self-loops to analyze long-time path concentration. This
small coined-walk experiment is not a reproduction of either construction.

## Glued binary trees: reproduce the transport behind an exponential separation

```bash
python examples/glued_trees.py --open
python examples/glued_trees.py --depths 2 4 8 16 32 --samples 129 --open
```

The command saves `.release-artifacts/glued-trees/glued_trees.html` and
`glued_trees.json`. The standalone report shows a small explicit graph, hidden
names and neighbor queries, column probabilities and complex amplitudes, final
exit probability curves, a depth sweep and separate CPU timings. `--output`
retains a separate run; `--seed` changes the bounded random-cycle reference
graphs; `--samples 17..257` changes inspection resolution, not the analytic
uniform-time success metric. Supply 1..8 distinct depths in 2..32.

The relevant construction is **two complete binary trees of depth n connected
at their leaves by one random alternating cycle**. It has `2^(n+2)-2` vertices.
All vertices except the entrance/exit have degree 3. Simply identifying the two
trees' leaves is a different graph: outperforming a random walker there does
not establish the stronger algorithmic result.

This experiment uses **continuous-time quantum walk**, not the four-port coined
maze walk. It follows `H = γA` with `γ=1/√2`. Starting at the entrance, column
symmetry is preserved despite the random weld. In the basis of normalized
equal-amplitude column states, H is a line of `2n+2` states with couplings 1
except for √2 across the central weld. NexQus executes an exact spectral
circuit: basis change, energy-phase evolution, inverse basis change. These
three logical reduced-register operations are not the paper's efficient
hidden-name oracle circuit or a hardware-native gate count.

For the classical continuous-time walk, each incident edge has jump rate γ.
The column probabilities obey a birth/death generator, with rates given by
the number of inter-column edges divided by the source column size. It is
not correct to square quantum column amplitudes and use those as transition
rates. A positive Poisson/uniformization calculation avoids cancellation of
the very small classical exit probabilities.

### Measurement protocol and recorded results

For each depth, set `T=4(n+1)` before evaluating the data. Reset to the entrance,
draw a final time uniformly in `[0,T]`, evolve without intermediate measurements,
then measure once. The quantum success probability is integrated analytically
over time; the classical average is computed with positive Poisson-tail weights.
The exit is **not absorbing**, and the curves show endpoint occupancy rather
than cumulative first detection. The sampled maximum is shown only as a
descriptive statistic, not as the time-selection strategy. This short window
is a numerical transport experiment, not the theorem's conservative guaranteed
polynomial-time window.

| Depth | Implied graph vertices | Columns simulated | Quantum uniform-time exit | Classical uniform-time exit |
| --- | ---: | ---: | ---: | ---: |
| 2 | 14 | 6 | 20.0378% | 5.00455% |
| 4 | 62 | 10 | 13.0539% | 0.946947% |
| 8 | 1,022 | 18 | 7.14511% | 0.0462481% |
| 16 | 262,142 | 34 | 3.73409% | 0.000148889% |
| 32 | 17,179,869,182 | 66 | 1.95686% | 0.00000000199371% |

The finite sweep shows a widening transport gap against the classical random
walk. It does not independently prove an asymptotic lower bound. A single
quantum attempt can fail: for example, depth 32 success is about 1.96%, not
near certainty. Independent resets/repeats can increase overall success.

The report also separates two expected model-time quantities: classical mean
first-passage time and repeated quantum uniform-time attempts with expected
evolution budget `T/(2p)`. This latter expression assumes zero reset/verification
cost and is not a QPU runtime estimate. At depth 2 it is approximately 29.94
model-time units versus 16.09 for the classical random walk: the quantum
protocol is worse on that metric at this small depth, even though its endpoint
success probability is higher. Never equate occupancy, first passage, oracle
queries and CPU milliseconds.

### What the theorem assumes, and what this demo computes

| Aspect | Exponential-separation setting | This local reproduction |
| --- | --- | --- |
| Graph access | Random hidden vertex names; local neighbor oracle | Known analytic column model; small graphs also explicit |
| Quantum evolution | Coherent oracle-based sparse Hamiltonian simulation | Spectral unitary on column-index qubits |
| Classical comparison | Lower bound against all classical algorithms in that oracle model | Exact random-walk probabilities and one bounded neighbor-query BFS |
| Large graph | Exponentially many vertices in n | Only 2n+2 column amplitudes computed |
| Conclusion | Published query-complexity theorem | Numerical transport reproduction and explainability |

At depth 32 the report implies a 17-billion-vertex graph, but it computes only
66 column states, padded into a seven-qubit column register. It does **not**
simulate 17 billion explicit vertices, a full 34-qubit compact vertex register,
or the theorem's 64-bit hidden-name register plus workspace. Both local models
benefit from knowing the columns; they are easy to simulate classically with
that information. The theorem concerns algorithms that are not given it.
The actual quantum dynamics do not need the column assignment; symmetry is
preserved automatically. The reduction is an analyst's simulation shortcut.
No paid QPU is used, and observing an exit name does not recover a route.

Full graphs at depths 2,3,4 are independently generated and checked. The
intertwining identity `Aγ B = B Hcolumn` verifies invariance, and full-vertex
spectral evolution agrees with SDK column states at five times. Tests cover
several random cycles/seeds. The bounded BFS receives only entrance name and
neighbor-name lists, identifies the other degree-2 root, and counts its queries.
It does not receive the exit name or columns. It is not a test of every
possible classical algorithm.

CPU medians use five measured trials after two warmups per method. They include
setup, the average-success calculation and one endpoint distribution at T/2,
but exclude graph generation, full-graph validation, plotting and export.
Differences reflect the numerical implementations of the two reduced models,
not a quantum computer running faster than a CPU.

Primary reference: [Childs et al., *Exponential algorithmic speedup by quantum
walk*, STOC 2003](https://arxiv.org/abs/quant-ph/0209131), Sections II, III C
(Eq. 27) and IV. The published exponential separation is specific to its
oracular access assumptions; it is not a blanket claim about ordinary mazes.

## Verification

### CPU and GPU circuit execution

Both walk examples accept `--device cpu`, `--device cuda` or `--device cuda:0`.
CUDA selects the optional Torch backend and fails if CUDA is unavailable.
Install a CUDA-enabled PyTorch build for your machine, then install the checkout:

```bash
python -m pip install -e .
python examples/glued_trees.py --device cuda --open
python examples/quantum_maze_walk.py --device cuda --steps 128 --open
```

Use `--backend torch --device cpu` to check the Torch path without CUDA.
Device selection applies to quantum circuit execution. Graph construction,
eigendecomposition, analytic success calculations, classical baselines and maze
exit-detector bookkeeping remain on CPU. Maze monitoring transfers each state
back to the host. Reports record the actual device and include transfers and
CUDA synchronization in wall times. These are hybrid execution paths; GPU
acceleration is not guaranteed for the small reduced registers.

The editable Grover maze example retains its CPU-only workflow. Existing saved
previews retain their original CPU measurements; generate a fresh report to
measure your selected device.

```bash
python -m unittest discover -s tests -p test_maze.py -v
python examples/quantum_maze.py --export .release-artifacts/maze/lesson.html
npm install --no-save --no-package-lock jsdom@26.1.0
node scripts/check_maze_ui.cjs
python examples/quantum_maze_walk.py
node scripts/check_maze_walk_ui.cjs
python -m unittest discover -s tests -p test_maze_walk.py -v
python examples/glued_trees.py
node scripts/check_glued_trees_ui.cjs
python -m unittest discover -s tests -p test_glued_trees.py -v
```

NumPy tests independently check phase flips, diffusion about the mean, the
analytic Grover probability formula, parity with the SDK's existing Grover
algorithm, seeded unfiltered measurement, maze paths and bounds. The UI test
also starts a real Python server and exercises editing, oracle rebuilding,
navigation, playback, stage measurement, downloads and no-path/budget cases.
jsdom is a development-only UI test dependency; students do not need Node.
Walk tests compare actual SDK states with an independent dense unitary and
exit projection, and compare the classical baseline with a position-only grid
Markov chain. They check locality, reversible shift, no wall leakage, total
probability accounting, no early arrival, unreachable exits and timing samples.
