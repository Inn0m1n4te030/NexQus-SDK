# Explainable QGANs and molecular energies

For rotatable atom/bond views, methane and larger organic examples, and H₂
geometry with calculated energy results, see [MOLECULES.md](MOLECULES.md).
`python examples/molecular_viewer.py` runs on native Windows without PySCF.

Unreleased source-checkout additions. The published PyPI 1.2.0 wheel does not
contain these APIs. Install this branch/checkout before using them:

```bash
python -m pip install -e ".[tutorial]"
python examples/qgan_distribution.py
```

This installs QGAN and both lessons 08/09 on native Windows, Linux or macOS.
Restart an already-running notebook kernel after installing.

For the PySCF molecular-integral driver and full H₂ bond-length example,
use **Linux, macOS, or WSL on Windows**. Add the chemistry dependency only
after the core/tutorial installation above succeeds:

```bash
python -m pip install -e ".[chemistry]"
python examples/h2_vqe.py
```

Both examples are entirely local. No AWS credentials, cloud simulator, QPU
submission or paid task is used. PySCF is an optional chemistry dependency;
core NexQus still depends only on NumPy. Other quantum SDKs are not required.
PySCF [does not support native Windows](https://pyscf.org/user/install.html).
Adding the chemistry extra on native Windows can cause pip to attempt an
unsupported source build and abort the whole combined installation before
updating NexQus. The optional chemistry integration is tested on Linux/Python
3.12; core/tutorial support does not imply PySCF platform support.

### Recover from a missing QGAN import

The message `Successfully built nexqus-sdk` means a wheel was built, not that
it was installed. If a combined install ends with `Failed to build pyscf`,
the kernel may still use an older NexQus package. A running kernel can also
retain the previous module until it is restarted. Both source and published
packages currently say 1.2.0, so the version string alone cannot identify them.

Run this **inside the notebook** to install the current source into its Python
environment, without PySCF or a local checkout path:

```python
%pip install --upgrade "nexqus-sdk[tutorial] @ https://github.com/Inn0m1n4te030/NexQus-SDK/archive/refs/heads/main.zip"
```

Then **restart the kernel**, Run All, and verify:

```python
import sys, nexqus
print(sys.executable)
print(nexqus.__file__)
from nexqus import QuantumDistribution, QGAN, Adam, distribution_metrics
```

For editable development, use `%pip install -e "/actual/path/NexQus-SDK[tutorial]"`
instead, pointing to an updated checkout. A normal installed source package can
also live in `site-packages`; the import path alone does not prove its version.

See the [measured three-seed and H₂ results](../benchmarks/results/qgan_h2_20261017.md)
for figures, final errors, dependency versions and comparison limitations.

## Discrete hybrid QGAN

```python
from nexqus import Circuit, Parameter, QuantumDistribution, QGAN

generator = QuantumDistribution(Circuit(1).ry(Parameter('theta'), 0))
model = QGAN(generator, {'theta': 1.3})
model.fit([.25, .75], epochs=160)
print(model.probabilities())
probabilities, jacobian = generator.value_and_grad(model.parameters)
print(jacobian['theta'])  # local changes in output probabilities per radian
generator.template.explain(bindings=model.parameters).to_html(filename='generator.html')
```

The generator is a pure-state Born machine: outcome probability is the squared
amplitude magnitude. It becomes a GAN through adversarial training with the
classical discriminator, not through a separate random latent-input vector.
The discriminator has one real logit per category; this simple lookup-table
architecture does not generalize across categories or encode molecular rules.
Targets are exact probability vectors, which can also represent empirical
histograms prepared by the caller. No train/test dataset split is implied.

For target probabilities t, generated probabilities p and discriminator
D = sigmoid(logits), the objectives are:

- D: `-0.5 * sum(t*log(D) + p*log(1-D))`.
- G: `-sum(p*log(D))`, with D held fixed during the generator update.

Default training uses five D updates (Adam, lr=.05) per G update (Adam,
lr=.02). `fit` accepts `generator_optimizer`, `discriminator_optimizer` and
positive `discriminator_steps`. GAN training can oscillate; more epochs do
not guarantee a lower final error. `history` records post-update objective
values, total variation and Jensen–Shannon divergence in natural-log units.
Calling `fit` again retains the model and discriminator weights, clears history
and creates fresh default optimizers unless the caller supplies optimizer objects.

`QuantumDistribution(template, max_qubits=12)` snapshots a static pure-state
circuit. Its Jacobian shifts individual supported parameterized gates and sums
their arithmetic chain rules, including repeated/shared parameters. It is an
exact local NumPy calculation, not sampled/hardware training or Torch autograd.
All 2^n outcome probabilities are stored. `sample(bindings, shots=1000, seed=None)`
returns integer basis indices; use n-bit formatting for q0-first labels.

`distribution_metrics(target, generated)` returns total variation (0..1) and
Jensen–Shannon divergence (0..ln(2)), handling zero-probability categories.
Probability vectors must already sum to one; malformed inputs are rejected.

### Classical comparison and explanations

The three-qubit example uses seeds 7, 21 and 42 and retains every result. The
classical generator is a trainable categorical softmax. Both methods start with
exactly the same probabilities and zero discriminator logits and use identical
objectives, learning rates and update counts. Generator parameterizations differ:
9 circuit angles versus 8 softmax logits (7 effective degrees of freedom);
execution costs are not matched, so this is not a speed or resource comparison.
The baseline is deliberately simple, not the strongest classical distribution
estimator; a histogram can represent the supplied target directly.

Outputs in `.release-artifacts/qgan/`:

- `comparison.png` / `.svg`: target/quantum/classical distributions, median and
  min–max total variation across seeds, and a probability-Jacobian heatmap.
- `generator.html`: gate-by-gate state explorer for the first seed's final model.
- `metrics.json`: every seed, full training histories, final distributions,
  hyperparameters, sensitivity arrays, dependency versions and methodology.

Jacobian entries are **local sensitivity**, not global causal attribution.
Training updates change parameters; gate navigation changes the execution
prefix at fixed parameters. These are different sequences and are labelled so.

## Weighted Hamiltonians and VQE

`PauliHamiltonian(n_qubits, terms)` takes `(real_coefficient, pauli_string)`
pairs in q0-first order. Duplicate strings combine and identity terms represent
constant energy offsets. `expectation(result)` needs a retained state.
`function(circuit)` returns a `HamiltonianFunction` with `__call__`,
`value_and_grad(bindings, wrt=None)` and `gradient`. This implementation evaluates
terms separately; it does not claim optimized measurement grouping.
`dense(max_qubits=10)` is a bounded exponential-size reference, not the VQE path.

```python
from nexqus import Parameter, Adam
from nexqus.chemistry import molecular_hamiltonian, h2_pair_ansatz

problem = molecular_hamiltonian(['H', 'H'], [[0, 0, 0], [0, 0, .735]])
circuit = h2_pair_ansatz(Parameter('theta'))
energy = problem.hamiltonian.function(circuit)
parameters = {'theta': 0.}; optimizer = Adam(.04)
for _ in range(180):
    value, gradient = energy.value_and_grad(parameters)
    parameters = optimizer.step(parameters, gradient)
print('VQE:', energy(parameters), 'Reference:', problem.exact_energy())
```

### Chemistry conventions and boundaries

`molecular_hamiltonian(symbols, coordinates_angstrom, basis='sto-3g', charge=0,
spin=0, max_qubits=12)` uses optional PySCF RHF to obtain real molecular-orbital
integrals. Only closed-shell systems (spin=0) are supported in this milestone.
Qubit preflight runs after basis setup but before SCF/integral transformation.
This is an all-orbital model; active spaces, frozen cores, relativistic/ECP
treatments, excited states and solvent models are not implemented.

`jordan_wigner(one_body, two_body, nuclear_repulsion=0., max_qubits=12,
tolerance=1e-12)` is available without PySCF for caller-supplied real spatial
integrals. Two-electron integrals use chemists' `(pq|rs)` order. Modes 2p/2p+1
are the alpha/beta spin orbitals; occupation 1 means an electron is present.
The mapping uses Pauli algebra, not a dense fermionic operator, and includes the
nuclear energy as an identity coefficient. The tolerance drops only final
mapped coefficients, so it is an approximation knob, not a chemistry accuracy
guarantee. Mapping, classical integral generation and simulation remain costly
as the number of orbitals increases; the qubit cap is not a process-RAM limit.

`MolecularProblem` records the Hamiltonian, electron count, RHF total energy,
nuclear repulsion, geometry and basis. `hartree_fock_circuit()` prepares the
occupied interleaved modes. `exact_energy(max_qubits=10)` diagonalizes the
Hamiltonian in the correct alpha/beta electron sector; it is a tiny-system
reference, not a scalable replacement for classical electronic-structure tools.

`h2_pair_ansatz(theta)` outputs
`cos(theta/2)|1100> + sin(theta/2)|0011>`. It is specialized to the H₂/STO-3G
two-determinant ground-state family, not a general chemistry excitation gate.
Final states have two electrons; intermediate decomposition states may not.
Gate-prefix energies are **not** a physical reaction path or VQE training curve.
The explorer's ZZII readout is a Pauli expectation, not total molecular energy.

Outputs in `.release-artifacts/h2/`:

- `comparison.png` / `.svg`: bond-length curves for RHF, independent PySCF FCI
  and NexQus VQE; optimization history; individual energy-term contributions.
- `circuit.html`: fixed-geometry circuit-state explanation.
- `metrics.json`: geometry, conventions, energies, Hamiltonian coefficients,
  optimization histories, term contributions and prefix particle-number checks.

All compared energies use the same basis/geometry and include nuclear
repulsion. The example checks the mapped exact energy against PySCF FCI to
1e-8 Hartree and final VQE to 1e-6 Hartree at every reported bond length.
This numerical agreement validates the small model, not drug efficacy.

## Validation and next steps

```bash
python -m unittest discover -s tests -p test_generative.py -v
python -m unittest discover -s tests -p test_hamiltonian.py -v
python -m unittest discover -s tests/chemistry -v  # chemistry extra required
```

Core tests compare probability/energy gradients against finite differences and
Jordan–Wigner output against independent occupation-basis fermionic matrices.
The optional integration suite compares H₂ at three bond lengths against PySCF
FCI and verifies Hartree–Fock energies and preflight rejection. CI also runs both
visual examples and uploads their complete outputs. Lessons 08/09 introduce
QGANs and a provenance-labelled fixed H₂ Hamiltonian without requiring PySCF.

Next chemistry work: active spaces, general electron-preserving ansätze and
larger-molecule validation. Molecular generation additionally needs a valid
chemical representation and validity checks; this discrete QGAN does not
generate molecules. Drug-discovery screening and biological validation are
separate future layers, not features delivered by this milestone.

References: [PySCF quickstart](https://pyscf.org/quickstart.html),
[PySCF FCI](https://pyscf.org/user/ci.html),
[PySCF integral transformations](https://pyscf.org/pyscf_api_docs/pyscf.ao2mo.html),
[PennyLane QGAN architecture example](https://pennylane.ai/demos/tutorial_quantum_gans).
The implementations here use NexQus circuits/simulation rather than those SDKs.
