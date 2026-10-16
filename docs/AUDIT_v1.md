# Public v1.0 readiness audit

Decision: **prepare/test 1.0.0rc1; do not publish stable 1.0.0 yet.**

The SDK has a coherent educational and small-research feature set. All six
original roadmap priorities are implemented. Review found real defects and
release omissions, so passing the earlier examples was not sufficient for a
stable public release decision. The confirmed defects are fixed in this
candidate and protected by new independent/reference regressions.

## Confirmed issues fixed

| Finding in v0.6 | Fix and evidence |
|---|---|
| NaN retained state could produce check='passed'; missing/invalid retained state and classical probabilities were incompletely checked | Validate finite values, exact shapes, normalization, state presence/representation and classical distributions before differences; regression tests |
| Unseeded trajectories could be called paired | Require an explicit seed as well as portable matching configuration |
| Nonunitary diagonal amplitudes 1+1e-6 and initial norm errors could be accepted | Explicit tight tolerances, shared normalized-state validation, roundoff removal; static/dynamic/QPE tests |
| Dynamic sampling allocated a host counter for every trajectory even with zero shots | Sparse shot schedule; guarded regression prevents a trillion-history run from allocating an 8 TB schedule before evolution |
| NumPy damping trajectories promoted single-precision state batches to complex128 | Preserve complex64 scalar arithmetic and verify internal state dtype in fast and generic noise paths |
| Complex learning data could lose imaginary values during conversion | Explicit real-data validation |
| A callback could mutate stored training-history snapshots | Independent callback parameter dictionaries |
| Adam could retain partial/nonfinite internal state after overflow | Validate all new moments/updates before atomic commit |
| A QuantumFunction template could gain unsupported dynamic operations after construction | Revalidate before evaluation/differentiation |
| Notebook engine note was misleading, stale outputs could survive a failed run, timeout reports were incomplete, and an old kernel could appear valid | Engine-specific reporting, cleared outputs, explicit timeout failure and per-notebook SDK-version validation |
| Minimal project metadata; no public distribution checks, source manifest, API policy or release workflow | Complete metadata, version single-source, source/wheel checks, CI, API reference/policy and public release procedure |

133 tests are now discovered. The local NumPy/Torch CPU run passes 127, with
the six CUDA tests explicitly skipped. Details and limitations are in VALIDATION.md
and the machine-readable reports. No test suite proves the absence of all bugs.

## Release blockers and ownership

| Gate | Current state | Next action |
|---|---|---|
| Public distribution terms | MIT approved by the maintainer on 2026-10-16 and applied | Complete |
| Public repository/support | Inn0m1n4te030/NexQus-SDK verified; issue tracking enabled | Source/Issues URLs applied; private reporting settings need owner access |
| PyPI identity/name | Maintainer confirmed ownership | Configure the separate TestPyPI account and index authentication |
| Candidate CUDA | Prior versions passed; changed candidate not run on CUDA here | Run check_release.py --require-gpu on the RTX 2070/target machine |
| Candidate real-kernel notebooks | Local kernel startup blocked by host socket restrictions | Run all seven with Jupyter and --require-gpu on a normal host |
| Cross-platform/version compatibility | CI matrix prepared, not executed in GitHub | Obtain a green run on the actual release commit |
| Index installation | No staged upload performed | Stage rc1 on TestPyPI, then test installation in a clean environment |

None of these gates requires redoing every earlier performance benchmark. Focus
on correctness of the changed candidate and the real installation path. The
public metadata check now has the approved MIT license and verified repository
URLs. Publication progress is recorded in RELEASE_STATUS.md.

## Recommended next work

1. Finish the release gates, let a few external users try rc1, fix their concrete
   installation/API/documentation issues, then tag 1.0.0 from the reviewed commit.
2. Make onboarding the first public product: a reliable pip install, working
   examples, searchable API docs, seven bilingual lessons and a clear bug-report
   path. Ask five learners/researchers to complete the first lesson independently.
3. Polish one practical reference application. A noisy VQE study with a classical
   exact solver, reproducible settings, error/energy plots and honest limits fits
   the current capabilities and can use generated/synthetic inputs with clear
   provenance. Choose a concrete task before expanding feature scope.
4. For the next minor release, prioritize features from observed demand: a Torch
   neural-network/autograd interface if QML integration is the need, or more
   scalable simulation kernels if circuit size/performance is the constraint.
   Measure end-to-end usefulness before promising quantum advantage.

Keep 1.0 focused: local circuit/noise/dynamic simulation, algorithms, analytic
learning foundations and education. Hardware services, distributed simulation,
custom fused kernels and automatic dynamic differentiation are future projects,
not prerequisites for a truthful first public SDK release.

The step-by-step release procedure, current primary PyPA references, validation
commands and owner decisions are in RELEASING.md. This audit does not establish
license/IP rights or account ownership; those cannot be inferred from passing code.
