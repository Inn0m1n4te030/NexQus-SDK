# NexQus public release procedure

## Current decision

Prepare and test **1.0.0rc1** first. Do not label v0.6 unchanged as a public stable
1.0: this audit found defects and missing distribution/maintenance work. The rc1
source fixes the confirmed defects and includes release automation. It is an
MIT-licensed public source candidate; index-upload status is recorded separately
in RELEASE_STATUS.md.

## Completed ownership decisions and account setup

The maintainer approved MIT licensing and confirmed PyPI ownership on
2026-10-16. LICENSE and the SPDX project.license value are now MIT. The verified
public source repository is https://github.com/Inn0m1n4te030/NexQus-SDK, with
issue tracking enabled and Source/Issues/Documentation metadata configured.

TestPyPI and PyPI authentication are separate from GitHub access. Complete the
following account setup before the authenticated upload:

1. Sign in to the intended TestPyPI account and confirm its access to the project.
   Production PyPI ownership does not sign in or grant access on TestPyPI.
2. Protect the repository's main branch and set GitHub environments named
   testpypi and pypi. Configure the appropriate repository, publish.yml workflow
   and environment as Trusted Publishers on the two indexes. Require review for
   the production environment. Configure private vulnerability reporting on the
   source repository. Never put account credentials in source or notebook files.

The public metadata check now validates the approved MIT license and real
repository links. It does not claim an authenticated index upload occurred.

## Validate the exact candidate

Use a fresh environment from the extracted candidate directory. Keep your
working CUDA-enabled PyTorch build; the SDK installation need not replace it.

```bash
python -m pip install --upgrade -e '.[tutorial,release]'
python -m ipykernel install --user --name nexqus --display-name "Python (NexQus)"
python scripts/check_release.py --require-gpu
python scripts/run_notebooks.py --engine jupyter --kernel nexqus --require-gpu
```

Release expectation: all 133 tests pass with no skips on the CUDA machine; all
seven notebooks pass, and the optional CUDA cell must execute. The previous
111-test v0.5 run and v0.6 Jupyter/GPU results do not validate newly changed code.
Use --report to preserve differently named evidence files. CUDA is intentionally
not claimed by ordinary hosted CI. Run it on the maintainer's trusted GPU machine;
do not expose a private self-hosted GPU runner to arbitrary pull requests.

On a Torch CPU machine use --require-torch. A NumPy-only validation can run with
no device flag; any optional skips remain visible. Missing CUDA is a failure
when --require-gpu is supplied.

Push the reviewed candidate source and obtain a green CI run. The supplied
ci.yml defines Python 3.10–3.14 on Linux/macOS/Windows, a NumPy 1.24.4 floor check,
Torch CPU, real Jupyter execution and installed-distribution checks. Adding a
workflow is not evidence that the matrix has run. Record the actual green commit.
The core dependency audit queries current advisory data. Audit the resolved
optional Torch/tutorial environment as well before public distribution, and
resolve collection errors or vulnerability findings instead of treating skipped
packages as clean. The local advisory-service connection timed out in this audit.

## Build and inspect

Begin with an empty dist directory (move old artifacts aside rather than upload
every historical file). From the source root:

```bash
python scripts/check_release.py --public
python -m build
python -m twine check --strict dist/*
python scripts/check_distribution.py
```

The distribution checker requires exactly one wheel and one sdist. It checks
metadata and included files, rebuilds a wheel from the extracted sdist, compares
runtime sources, installs both wheels in separate fresh virtual environments,
and runs the suite from outside the source checkout. Its JSON report includes
artifact SHA-256 hashes. Build and validation perform no uploads.

Editable installation alone does not verify what a PyPI customer receives.
The source distribution includes examples, tests, notebooks and scripts. The
wheel intentionally contains runtime modules and metadata only.

## Stage on TestPyPI

After account setup and the required validation are complete, manually run the
Publish reviewed distribution workflow from the
reviewed commit, choose testpypi, and enter the exact source version 1.0.0rc1.
The workflow reruns CI and distribution checks, enforces public metadata, then
uses the configured TestPyPI environment and Trusted Publisher.

For a manual upload instead, use a local token prompt/keyring; never paste the
token in a chat, script, notebook or shell command argument:

```bash
python -m twine upload --repository testpypi dist/*
```

Verify the TestPyPI page and install that exact prerelease in a new environment.
Install dependencies from the normal index first, then retrieve only NexQus from
TestPyPI. This avoids mixing indexes for dependency selection:

```bash
python -m pip install 'numpy>=1.24,<3'
python -m pip install --no-deps --index-url https://test.pypi.org/simple/ nexqus-sdk==1.0.0rc1
python -c "import nexqus; print(nexqus.__version__)"
```

Run the Bell example, a noisy circuit, and the source test suite with that
installed distribution. Keep the version, hashes and validation records. Try the
README instructions with at least a few external beginners before the stable tag.

## Promote to stable 1.0.0

Freeze the API and scope. Update the single __version__ value to 1.0.0, the
Development Status classifier to the intended stable status, changelog and README
release statements. Build new clean stable artifacts; prerelease files cannot be
renamed into a stable version. Run the release/distribution checks for these exact
artifacts and obtain green CI on the release commit. If runtime code changed
since rc validation, repeat affected GPU and notebook checks.

Tag the reviewed commit v1.0.0. Use the publish workflow with repository=pypi and
the exact version 1.0.0, with production-environment review. The workflow rejects
prerelease versions on its production path. Then check the real public project
page, install nexqus-sdk==1.0.0 from ordinary PyPI in a fresh environment, and run
smoke examples. Publish concise release notes linking to working docs and support.
Do not call the release published until the index shows the uploaded artifacts.

## Primary packaging references checked for this audit

- https://packaging.python.org/en/latest/tutorials/packaging-projects/
- https://packaging.python.org/en/latest/guides/writing-pyproject-toml/
- https://packaging.python.org/en/latest/guides/using-testpypi/
- https://docs.pypi.org/trusted-publishers/using-a-publisher/

The metadata uses current SPDX license-expression fields supported by
setuptools>=77.0.3. Trusted Publishing uses the PyPA action with job-level
id-token:write; test and production index identities are separate.
