# NexQus SDK 0.6.0 — 16 October 2026

Priority 6 delivery: a bilingual beginner notebook path. The v0.5 runtime engine
is retained; only its package version changes. This is a teaching/materials
release rather than another simulator optimization release.

## Included

- Seven independent notebooks, from first circuit to an end-to-end experiment.
- English explanations, Burmese summaries/comments, exercises and expected-result assertions.
- Five embedded plots and executed CPU outputs.
- Regenerable notebook source and an execution/reporting script.
- Optional Jupyter dependencies and kernel setup instructions.
- A clearly optional CUDA parity cell; a --require-gpu executor mode.

## Verification and boundaries

All 35 plain Python code cells executed successfully using the bundled
standalone runner in fresh per-notebook subprocesses. Outputs/plots are saved.
The optional CUDA cell was skipped because PyTorch is not installed here.
A Jupyter runtime was unavailable and could not be installed from this
environment's package index. Actual Jupyter-kernel execution remains unverified
here; an explicit nbclient/Jupyter execution route is supplied for the user.

The user previously confirmed all 111 v0.5 tests passed with no skips on their
CPU/GPU system. The retained suite locally passes 101 CPU tests, with ten Torch
tests skipped. Historical benchmark/learning logs refer to their versioned releases.

The original six priorities now have implementations/deliverables, with this
release's Jupyter-kernel and optional CUDA tutorial validation outstanding on
the target environment. This is not a claim of production readiness or
comprehensive algorithm/hardware validation.
