# Public API policy for NexQus 1.x

This policy covers the documented public API beginning with version 1.0.0.

## Supported surface

The top-level exports in nexqus.__all__, their documented public methods, and
algorithm functions listed in docs/API.md form the supported interface. The
documented compare_results and trajectory_confidence_bound functions are also
supported. q0-first/MSB bit ordering, radians, channel parameter conventions,
parameter binding, result fields and classical-key order are part of the contract.

Backend, DeviceRandom, Expression internals, Operation construction, private
helpers, direct mutation of Circuit.operations, simulator/backend configuration
attributes and undocumented metadata fields are implementation details. Build
circuits through public methods and create a new Simulator to change its backend,
precision or method. Do not alter a template during evaluation/training.

## Versioning

- Patch releases fix defects and documentation while preserving valid documented
  use. Rejecting inputs that violate physical or documented constraints is a fix.
- Minor releases may add gates, channels, methods, optional fields and examples.
  Readers of metadata must tolerate additional fields.
- Deliberately incompatible supported-API changes require a major version.
  When feasible, removal is preceded by a warning in at least one minor release.

Bug fixes may change previously incorrect numerical or stochastic results.
Exact bitwise identity across versions, platforms, precision, RNG engines, batch
sizes, thread settings or numerical-library versions is not promised.
Use precision-appropriate tolerances for physical results.

## Reproducibility

Record SDK/Python/NumPy/Torch/CUDA versions, device, precision, circuit, method,
noise, seed, trajectory count, batch size, RNG engine and fast-path options.
Portable RNG provides paired device streams for a fixed configuration; native
RNG does not provide cross-backend paired streams. The portable simulation PRNG
is noncryptographic and its independence assumptions are not a rigorous PRNG
certification. A supplied seed is required for paired trajectory validation.

Compiled circuits snapshot the source circuit and noise and belong to their
creating simulator. They prepare execution tensors; they are not native code or
kernel fusion. Sweeps execute sequentially in binding order and use reproducible
child seeds; streaming sweeps avoid accumulating every result.

## Supported environments and limits

The package declares Python >=3.10 and NumPy >=1.24,<3; PyTorch >=2.2,<3 is optional.
The CI matrix tests selected current versions, not every possible combination.
Consult the repository CI results for versions actually executed. CPU and NVIDIA CUDA are the supported
device families. There is no implicit fallback from an unavailable device.

An estimated memory guard is not a hard allocator limit. No performance SLA,
fixed maximum qubit count, hardware fidelity or quantum advantage is promised.
Sampled trajectories are approximate even when final shots are disabled.
