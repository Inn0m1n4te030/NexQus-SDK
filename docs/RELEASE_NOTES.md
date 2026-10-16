# NexQus SDK 1.0.0

NexQus 1.0 provides a Python quantum circuit simulator for education and small
research experiments, with NumPy CPU and optional PyTorch CPU/NVIDIA CUDA execution.
The package is distributed under the MIT license.

## Install

```bash
python -m pip install nexqus-sdk==1.0.0
```

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
