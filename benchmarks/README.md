# Independent simulator benchmarks

The comparison tools live outside the NexQus runtime. Install optional
development dependencies from this checkout using Python 3.11+ (CI uses 3.12).
The NexQus runtime continues to support Python 3.10+:

```bash
python -m pip install -e '.[benchmark-compare]'
python benchmarks/run.py --suite smoke
python benchmarks/run.py --suite correctness
python benchmarks/run.py --suite standard --threads 1 --warmup 3 --repeats 9 --output benchmarks/results/local
```

The standard suite's [measured CPU report](results/cpu_20261017/report.md) includes
raw JSON and CSV with slower NexQus workloads retained. Correctness evidence is
in [the small-circuit report](results/correctness_20261017/report.md).
The [optimization comparison](results/optimization_20261017.md) compares the
unoptimized and optimized SDK using the same corrected timing protocol. The
[unoptimized baseline](results/baseline_cpu_20261017/report.md) is a fresh rerun,
not the earlier report with memory sampling active during timing.

Correctness and smoke suites contain **no execution timing fields** in their
JSON, CSV or Markdown output. They verify numerical results. Only the standard
suite measures performance; it requires at least two equal warmups and three
timed repetitions for every engine. Defaults are three warmups and nine repeats.

## Identical workloads

`cases.py` builds SDK-independent, seeded gate specifications. Each adapter
translates them independently, using the SDK's native gates. They do not reuse
NexQus matrices, state evolution, or templates. Workloads cover GHZ, asymmetric
rotation/entangling circuits, QFT with controlled phases and swaps, diagonal
circuits, and exact noisy density evolution. No terminal shots are sampled.
The correctness suite uses 45 circuits (2, 3, 5 qubits; three seeds; five families).
The standard suite includes 4/12/18-qubit statevectors and 4/8-qubit noisy densities.

All adapters start in the all-zero state and return the full CPU state and
probabilities in logical q0-first/MSB order. Qiskit maps logical wire i to
physical wire n-1-i. For depolarization all engines implement
`E(rho) = (1-p)rho + p I/2`: PennyLane's channel probability is explicitly `3p/4`.
Amplitude damping uses the same gamma in every SDK. Noise is applied on every
wire after each complete layer, rather than relying on different default noise
insertion rules. Full specifications and channel placements are in each JSON.

## Accuracy

Small circuits use an independent full-matrix complex128 oracle with explicit
bit-index embedding. Analytic reference tests cover asymmetric controls,
amplitude damping and depolarization. Larger runs compare to Qiskit Aer, with
the reference named in every result. Complex128 tolerances are 1e-9 for maximum
state error; complex64 uses 2e-5. Probability TVD and normalization limits are
ten times those values. Statevectors align global phase; density matrices
compare full entries. Returned probabilities must also match each returned
state. Finite entries, Hermiticity and small-density eigenvalues are checked.
Accuracy validation runs outside measured timings. A mismatch or missing
requested dependency fails the command; errors remain in the reports.

## Runtime and memory

Every engine/case runs sequentially in a fresh subprocess on the same host.
Backend order is seeded and shuffled. Thread environment variables are set
before imports; threadpoolctl and Aer options enforce the requested CPU count.
Actual thread pools, CPU model, affinity, OS, Python, package versions, source
hashes, seeds, precision and settings are captured. CPU affinity records allowed
cores; it does not pin the process to a single core.

Timing includes prepared execution and full state/probability materialization.
Imports, construction and preparation are excluded and reported together as
`preparation_seconds`. NexQus uses compile(), Aer uses optimization_level=0
transpilation, and PennyLane uses a prebuilt QuantumScript with default.qubit or
default.mixed. Fusion is disabled; Aer idle-wire truncation is disabled. These
settings compare the same elementary workloads and do not represent each SDK's
fastest optimized execution. API/result overhead is included, especially
visible on small circuits. Raw repetitions, median, range and standard deviation
are saved; timing is descriptive, not a CI speed threshold.
Shared NumPy random initialization happens equally before preparation for every
engine, and each engine completes the same number of untimed warmups. Framework
lazy initialization is therefore excluded from steady-state execution timing.
No memory sampler runs in the timed process.

RSS includes native and Python memory. A second fresh subprocess per engine/case
measures memory, with the same warmup count followed by an untimed execution.
RSS is sampled every 1 ms; shorter allocation peaks can be missed. A separate
OS high-water RSS includes that memory process's imports and setup.
Baseline RSS and incremental sampled peaks are included: a zero increment can
mean allocator reuse. State/probability byte sizes are reported separately.
These measurements are process memory, not GPU VRAM or isolated tensor usage.

Double precision is the common three-engine baseline. `--precision single`
supports NexQus and Aer; request `--engines nexqus qiskit` for that comparison.
PennyLane's default-device adapter explicitly reports single precision as
unsupported, rather than timing a promoted double-precision computation.

The current suite measures local CPU statevector and exact density simulation.
PennyLane Lightning, CUDA, trajectories, mid-circuit measurements, gradients,
hardware services and distributed execution need separate matched protocols.
Run on an otherwise idle host; shared/cloud CPU scheduling affects timings.
The recorded runtime/benchmark source hashes identify the code even when
`git_dirty` is true; `git_commit` then identifies the checkout's parent revision.
`requirements-cpu.txt` records the environment used for the published run.

API references: [AerSimulator](https://qiskit.github.io/qiskit-aer/stubs/qiskit_aer.AerSimulator.html),
[PennyLane devices](https://docs.pennylane.ai/en/stable/introduction/circuits.html),
[PennyLane depolarization](https://docs.pennylane.ai/en/stable/code/api/pennylane.DepolarizingChannel.html).

## Reproduce the unoptimized baseline

Extract the runtime from commit `89050b4e35194a2587ebceb4e64a6be80a0b89b2`
(`git archive COMMIT src`), then run this checkout's benchmark harness with
`PYTHONPATH` pointing to the extracted `src` directory. Run the optimized source
with the same Python environment, host, circuits, precision, threads, warmups
and repetitions. The reports record the imported runtime path and source hash.
The summary includes all workloads, including regressions or cases still slower
than another simulator. Earlier report versions remain in Git history.


## Convolutional image learning experiment

`python benchmarks/image_learning.py` reproduces the single-seed, CPU procedural
bars experiment in [these recorded results](results/conv_bars_20261018.json).
[Before/after images](../examples/convolutional_previews/report.html) show
conditional orientation learning and limited position diversity after 1,000
updates. This is a learning diagnostic, not a simulator performance comparison
or validation of pet-image quality or quantum advantage.
