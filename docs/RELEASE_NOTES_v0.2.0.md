# NexQus SDK 0.2.0 — 15 October 2026

## Additions

- Named real parameters with arithmetic, strict binding validation, symbolic
  inverse circuits and composition.
- Ordered seeded parameter sweeps and a streaming iterator; no retained state
  by default for sweeps.
- Prepared concrete circuit/noise snapshots bound to one simulator instance.
- Content-keyed LRU gate tensor cache, configurable budget, cache statistics
  and cache clearing. Equivalent gate matrices share backend tensors.
- Batched independent noisy trajectories, partial final batches, and a
  batch-aware working-memory estimate. Default batch size stays 1.
- CPU/GPU benchmark runner with warmup, repeated median times, environment
  metadata, NumPy probability reference checks and optional CUDA peak allocation.
- Strict GPU validation entry point that fails on unavailable CUDA or skipped
  tests, rather than reporting an incomplete validation as success.
- New examples: parameter sweeps, repeated prepared execution, batched noise.
- Expanded English and Burmese documentation.

## Compatibility and limits

The v0.1 gate/function API, default bit order and algorithms are preserved.
No mid-circuit measurement, reset/feed-forward, gradients, QNN training,
distributed simulation or hardware connectors are added. Prepared execution is
not native compilation. Parameter sweeps are sequential; trajectory batching is
the batched execution feature in this release.

The main state capacity does not increase exponentially with these optimizations.
Higher batch sizes use more memory and can reduce the maximum qubit count.
Changing trajectory batch size changes random draw ordering. Hardware speedups
must be benchmarked rather than promised.

## Validation record

- The user reported v0.1 CPU and GPU tests passed; complete GPU logs were not
  supplied, so no hardware-specific performance claim is inferred.
- v0.2 local suite: 48 tests discovered, 44 passed, 4 optional Torch tests skipped
  because PyTorch is not installed here.
- New tests cover parameter arithmetic, invalid bindings, sweep seeds,
  symbolic inverses, prepared snapshots, source mutation, cache limits,
  batching, partial batches, channel endpoints, custom Kraus channels,
  noisy convergence, memory guards and single precision.
- New Torch CPU/CUDA parity tests cover prepared/batched execution and sweeps.
  They must be run on the target machine for v0.2 GPU validation.
- CPU benchmark output and full local test log are bundled in this directory.

`RELEASE_NOTES.md`, `test_results.txt`, and `cpu_benchmark.json` record the original
v0.1 baseline; v0.2 records use filenames containing v02 or v0.2.0.
