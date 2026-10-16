# Contributing to NexQus

NexQus SDK is released under the MIT License. Report reproducible bugs and propose
changes through https://github.com/Inn0m1n4te030/NexQus-SDK. By contributing code,
you agree that your contribution is provided under the project's MIT License.

Install in a fresh environment with `python -m pip install -e .`. Run
`python scripts/check_release.py`. CPU-only installations may skip optional
Torch/CUDA tests, and reports must retain those skips. If Torch CPU is available,
run `python scripts/check_release.py --require-torch`. CUDA maintainers use
`--require-gpu`, which requires zero skips.

Changes to evolution, tensor ordering, stochastic sampling, precision, memory
guards or gradients need a regression based on a known answer or independent
reference. Include seeds, precision, method and batch size. An agreement between
two implementations sharing a kernel is not sufficient evidence on its own.

Update the API documentation and changelog when observable behavior changes.
Write clear English docstrings and preserve Burmese explanations. New notebooks
should be generated from scripts/build_notebooks.py and executed using a real
Jupyter kernel before release. Keep optional-device skips explicit.

The source distribution includes tests and teaching material. Before proposing a
release, run `python -m build`, `python -m twine check --strict dist/*`, and
`python scripts/check_distribution.py` from a clean dist directory.

For bug reports include a minimal complete circuit, Python/NumPy/Torch versions,
OS/device, expected result, actual result and simulator metadata. Remove private
data and credentials. Use https://github.com/Inn0m1n4te030/NexQus-SDK/issues.

The public API policy is in docs/API_STABILITY.md. Avoid depending on
private helpers, direct Operation construction or mutable implementation internals.
