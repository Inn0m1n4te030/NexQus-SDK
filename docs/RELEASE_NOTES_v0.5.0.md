# NexQus SDK 0.5.0 — 16 October 2026

Priority 5: mid-circuit projective measurement, reset and single-bit classical
feed-forward. Added exact branch-density and sampled dynamic engines,
classical/joint result fields, reduced density matrices, protocol composition
with classical-key renaming, and a reusable teleportation circuit.

Static simulation and v0.4 learning remain available. Dynamic gradients,
compound conditions, conditional measurement, pulse/hardware controls and
specialized dynamic noise performance kernels are not implemented.

## Validation record

- User confirmed v0.4 strict GPU validation passed.
- v0.5 local suite: 111 discovered, 101 passed, 10 optional Torch CPU/CUDA tests
  skipped because PyTorch is not installed here.
- Teleportation fidelity in the example: 0.9999999999999997.
- Protocol tests cover 12 arbitrary complex input states and verify removing
  corrections destroys the recovered state.
- Other checks cover collapse, reset, classical overwrites, conditional noise,
  joint samples, sampling convergence, snapshots, key remapping and memory guards.
- New dynamic CUDA paths must be checked with examples/validate_gpu.py.

See V05_GUIDE.md for exact/sampled semantics and limits. Historical guides and
logs retain versioned names. Tested beginner notebooks remain the next
uncompleted item from the original roadmap.
