# Releasing NexQus SDK

This guide is for maintainers. Public changes belong in CHANGELOG.md and
RELEASE_NOTES.md. Generated test reports and audit outputs belong in
.release-artifacts/ or CI artifacts; keep them out of the source distribution.

## Prepare a release

The current candidate is **1.3.0rc1**, targeting stable **1.3.0**. Candidate
artifacts have not been uploaded. Validate the checkout/wheels and collect
volunteer feedback first; only TestPyPI accepts candidates in this repository's
publishing workflow. Production promotion requires a final version bump to
1.3.0, regenerated notebooks, repeated package validation and reviewed CI.
Do not copy historical CPU reports as evidence of a CUDA hardware run.

1. Update the single __version__ value in src/nexqus/__init__.py.
2. Update the changelog, release notes and installation instructions, including
   notebooks/README.md and the setup/version text in scripts/build_notebooks.py.
   Regenerate all notebooks with python scripts/build_notebooks.py, then execute
   them for validation. Keep generated source notebooks free of saved outputs.
3. Review API compatibility, license metadata, documentation links and package contents.
4. Obtain passing CI on the reviewed commit. CI covers Python 3.10–3.14 on
   Linux/macOS/Windows, minimum NumPy, Torch CPU, Jupyter, installed distributions
   and independent simulator correctness. Optional comparison tools use Python 3.11+.

## Validate

In an activated environment from the source checkout:

```bash
python -m pip install -e '.[tutorial,release,visualization,benchmark-compare]'
python scripts/check_release.py --public
python -m ipykernel install --user --name nexqus --display-name 'Python (NexQus)'
python scripts/run_notebooks.py --engine jupyter --kernel nexqus
python benchmarks/run.py --suite smoke
```

On a compatible CUDA machine, keep the intended CUDA-enabled PyTorch build and run:

```bash
bash scripts/validate_cuda_release.sh
```

The strict CUDA check requires zero skipped tests and executes the optional GPU
notebook cell. Repeat affected CUDA checks when simulation code changes. Record
SDK/dependency/device versions and audit the resolved release environment.

## Build and inspect

Start with an empty dist directory so historical files cannot be uploaded by accident:

```bash
python -m build
python -m twine check --strict dist/*
python scripts/check_distribution.py
```

The distribution checker verifies public documentation contents, compares runtime
sources, rebuilds from the sdist, installs both wheels in fresh environments and
runs tests outside the source checkout. Reports go to .release-artifacts/.
The runtime wheel contains importable modules and metadata. The sdist also
contains public guides, examples, notebooks, tests and maintenance scripts.

## Publish

Configure separate TestPyPI and PyPI Trusted Publishers for repository
Inn0m1n4te030/NexQus-SDK, workflow publish.yml, and environments testpypi/pypi.
Restrict production publishing through the pypi environment's protection rules.
Use the manual Publish reviewed distribution workflow from the reviewed commit,
choose the target index and supply the exact source version. CI and distribution
checks run before upload. Production rejects prerelease versions.

TestPyPI is suitable for release candidates. Install dependencies from normal PyPI
first, then retrieve only the SDK from TestPyPI with --no-deps. After a production
upload, verify a fresh installation from ordinary PyPI and run smoke examples:

```bash
python -m pip install nexqus-sdk==1.3.0
python -c 'import nexqus; print(nexqus.__version__)'
```

Confirm the published file hashes against the publishing job and create the
matching release tag. Announce a release only after the index upload and fresh
installation have succeeded. Never put credentials in source, logs or notebooks.
