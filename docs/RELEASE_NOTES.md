# NexQus SDK 1.1.0

## Unreleased source-checkout additions

Numerical circuit angle labels, logical depth/counts, optional PNG/PDF export,
reusable QML templates and independent simulator benchmarks are available in
the development checkout. See [API.md](API.md), [QML_TEMPLATES.md](QML_TEMPLATES.md)
and [the benchmark protocol](../benchmarks/README.md). They have not been added
to the existing PyPI 1.1.0 archives. No publication is triggered by these docs.

NexQus provides a Python quantum circuit simulator for education and small
research experiments, with NumPy CPU and optional PyTorch CPU/NVIDIA CUDA execution.
The package is distributed under the MIT license.

## Install

```bash
python -m pip install nexqus-sdk==1.1.0
```

## New in 1.1

- Native text/SVG circuit drawing, Jupyter display, file export and wrapping,
  including custom circuits, idle wires, measurements and classical conditions.
- Shor factoring and quantum order finding with configurable counting precision,
  resource estimates and memory checks; quantum-path examples through 77.
- Compact basis-permutation operations across NumPy/PyTorch simulation modes,
  compilation, inverses, noise and classical conditions.

Shor uses logical modular permutation gates, not an elementary-gate hardware
arithmetic decomposition. Statevector storage still grows exponentially in total
qubits. Memory estimates are not hard process-RAM caps. Circuit drawings show
stored gate names and symbolic expressions; concrete original angles are not
retained by the current circuit representation.

## Included features

- Statevector, exact density-matrix and batched noisy trajectory simulation.
- Reusable circuit functions, symbolic parameters, binding, sweeps and prepared execution.
- Built-in algorithms including Bell/GHZ, QFT, Grover, phase estimation and teleportation.
- Parameter-shift gradients, Adam/SGD, regression, binary classification and VQE examples.
- Mid-circuit measurement, reset, single-bit feed-forward and reduced density matrices.
- Seven English/Burmese teaching notebooks and a Burmese quick start.

## Compatibility and limits

Python 3.10+ and NumPy >=1.24,<3 are required. PyTorch >=2.2,<3 is optional;
choose a build compatible with the intended CPU/CUDA environment.

The public 1.x compatibility contract is described in [API_STABILITY.md](API_STABILITY.md).
Qubit capacity depends on memory, simulation method, precision and trajectory batch size.
Sampled trajectories are approximate. Dynamic-circuit gradients, distributed execution,
hardware services and pulse simulation are outside this release's scope.

See [API.md](API.md), [LEARNING.md](LEARNING.md), [PERFORMANCE.md](PERFORMANCE.md)
and [DYNAMIC_CIRCUITS.md](DYNAMIC_CIRCUITS.md) for supported usage and conventions.
