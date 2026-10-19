# NexQus SDK 1.2.0

The source image-GAN workflow offers opt-in regularized training for both
classical and hybrid models, a fresh `dcgan` architecture, checkpoint retuning
without discarding learned weights, and generated-images-first reports with
duplicate/latent diagnostics. Pet-image quality is still unvalidated; see
[the image guide](IMAGE_GANS.md) for bounded retuning and fresh-run commands.


## Unreleased source-checkout addition

The [Quantum Maze Explorer](QUANTUM_MAZE.md) is a local school-project example:
draw mazes, inspect a bounded Grover search and BFS baseline, measure any stage,
and export an offline circuit lesson. It requires only core NumPy. Classical
oracle construction is disclosed; it does not establish a maze-solving speedup.

The image example now defaults to convolutional classical/hybrid decoders and
an image/label-conditioned convolutional discriminator. Epoch budgets,
training-only horizontal flips, seeded file selection and periodic fixed-noise
previews/checkpoints support longer experiments. Old MLP checkpoints remain
loadable. The recorded procedural-bars experiment shows some conditional
structure with limited diversity; recognizable pet generation remains unvalidated.

The [GPU workflow guide](GPU.md) covers shared device selection, native result
tensors, batched differentiable Torch circuits, image GANs, QGAN/learning and
Hamiltonian/VQE/exact references. Optional GPU4PySCF dispatch accelerates RHF;
integral generation/mapping remain CPU. CPU parity tests do not establish CUDA
execution: the new strict hardware validator requires a real GPU and zero skips.

Two [conditional image QGANs](IMAGE_GANS.md) add quantum pixel patches and
quantum features with a classical decoder, a matched-update classical GAN,
folder/optional CIFAR-10 loading, minibatch training, exact checkpoint
continuation and parameter-to-pixel explanations. Lesson 11 runs locally on
CPU with optional PyTorch, including native Windows. No pretrained pet model
or QPU execution is included. Core installation remains NumPy-only.

The [native molecular viewer](MOLECULES.md) adds rotatable atom/bond structures,
distance/angle measurements, SVG/XYZ/JSON export and H₂ geometry linked to VQE
energies. Methane, water, ethene and benzene geometry examples work on native
Windows without PySCF. Lessons 09/10 show the structure and the calculation.

Local [QGAN and H₂ chemistry tools](QGAN_CHEMISTRY.md) add probability gradients,
hybrid adversarial training with a classical baseline, probability sensitivities,
weighted Hamiltonians, optional PySCF integrals/Jordan–Wigner mapping and
classical-reference-checked VQE examples. Lessons 08/09 explain both workflows.
These are not included in the published 1.2.0 wheel. General molecular generation,
active-space chemistry and drug-discovery screening remain future work.

The local [circuit explainability explorer](EXPLAINABILITY.md) adds exact
gate-by-gate state traces, offline HTML visualization, QNN block/parameter
annotations and Bell/interference/trained-QNN examples. It is not included in
the published 1.2.0 wheel. It supports static pure-state simulation; feature
attribution and noisy/dynamic explanations are future work.

The optional [Amazon Braket bridge](BRAKET.md) adds offline preparation, local
simulation, explicit paid submission and job/result handling. It is not included
in the published 1.2.0 wheel. Tests use local simulation and mocked AWS calls;
no cloud task has been submitted and actual QPU execution is unverified.

## New in 1.2

- Visual quantum-versus-classical sine regression with prediction/loss/error
  plots, held-out scores across three seeds and a matched sinusoidal baseline.
- Numerical RX/RY/RZ/P/CP angle labels, logical circuit depth and operation/gate counts.
- Optional PNG/PDF export from the native SVG renderer through CairoSVG/native Cairo.
- Basis/angle encoding, configurable entangling and variational layers, data
  reuploading and a new bilingual tutorial. Helpers support symbolic parameters
  and the existing analytic gradients.
- Independent correctness and CPU performance comparisons against optional
  Qiskit Aer and PennyLane development dependencies. Correctness reports contain
  accuracy only; performance uses equal warmups and separate memory sampling.
  Reports retain workloads where NexQus is slower.
- Faster NumPy single-wire kernels and exact Pauli/damping channels, compiled
  channel recognition and no RNG setup for unseeded exact zero-shot runs.

See [API.md](API.md), [QML_TEMPLATES.md](QML_TEMPLATES.md),
[the benchmark protocol](../benchmarks/README.md) and
[measured before/after results](../benchmarks/results/optimization_20261017.md).
The recorded benchmark runs identify their original runtime source hashes and
pre-release version label; those historical measurements are preserved unchanged.

NexQus provides a Python quantum circuit simulator for education and small
research experiments, with NumPy CPU and optional PyTorch CPU/NVIDIA CUDA execution.
The package is distributed under the MIT license.

## Install

```bash
python -m pip install nexqus-sdk==1.2.0
```

For tutorials and PNG/PDF export:

```bash
python -m pip install 'nexqus-sdk[tutorial,visualization]==1.2.0'
```

Native Cairo must also be available for PNG/PDF conversion. Text/SVG and QML
templates require no optional runtime dependencies. Simulator comparisons need
Python 3.11+ and the `benchmark-compare` extra in a source checkout.

## Retained from 1.1

- Native text/SVG circuit drawing, Jupyter display, file export and wrapping,
  including custom circuits, idle wires, measurements and classical conditions.
- Shor factoring and quantum order finding with configurable counting precision,
  resource estimates and memory checks; quantum-path examples through 77.
- Compact basis-permutation operations across NumPy/PyTorch simulation modes,
  compilation, inverses, noise and classical conditions.

Shor uses logical modular permutation gates, not an elementary-gate hardware
arithmetic decomposition. Statevector storage still grows exponentially in total
qubits. Memory estimates are not hard process-RAM caps. Circuit drawings now
retain numerical angles for supported built-in gates as well as symbolic expressions.

## Included features

- Statevector, exact density-matrix and batched noisy trajectory simulation.
- Reusable circuit functions, symbolic parameters, binding, sweeps and prepared execution.
- Built-in algorithms including Bell/GHZ, QFT, Grover, phase estimation and teleportation.
- Parameter-shift gradients, Adam/SGD, regression, binary classification and VQE examples.
- Mid-circuit measurement, reset, single-bit feed-forward and reduced density matrices.
- Eight English/Burmese teaching notebooks and a Burmese quick start.

## Compatibility and limits

Python 3.10+ and NumPy >=1.24,<3 are required. PyTorch >=2.2,<3 is optional;
choose a build compatible with the intended CPU/CUDA environment.

The public 1.x compatibility contract is described in [API_STABILITY.md](API_STABILITY.md).
Qubit capacity depends on memory, simulation method, precision and trajectory batch size.
Sampled trajectories are approximate. Dynamic-circuit gradients, distributed execution,
hardware services and pulse simulation are outside this release's scope.

See [API.md](API.md), [LEARNING.md](LEARNING.md), [PERFORMANCE.md](PERFORMANCE.md)
and [DYNAMIC_CIRCUITS.md](DYNAMIC_CIRCUITS.md) for supported usage and conventions.
