"""H2/STO-3G energy curve, local NexQus VQE vs PySCF RHF and FCI.

Run: python examples/h2_vqe.py (requires chemistry,tutorial extras).
The reference and VQE use the same geometry, basis and electron sector.
All energies include nuclear repulsion. No QPU or cloud calls.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import numpy as np
import matplotlib.pyplot as plt
import pyscf
from pyscf import gto, scf, fci
import nexqus
from nexqus import Parameter, Adam, Simulator
from nexqus.chemistry import molecular_hamiltonian, h2_pair_ansatz


def generate(output, distances=(.5, .735, 1., 1.5, 2.)):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    rows = []; detail = None
    for distance in distances:
        xyz = [[0, 0, 0], [0, 0, float(distance)]]
        problem = molecular_hamiltonian(['H', 'H'], xyz)
        mol = gto.M(atom=list(zip(['H', 'H'], xyz)), basis='sto-3g', unit='Angstrom', verbose=0)
        rhf = scf.RHF(mol).run(conv_tol=1e-11)
        reference = float(fci.FCI(rhf).kernel()[0])
        circuit = h2_pair_ansatz(Parameter('theta')); function = problem.hamiltonian.function(circuit)
        parameters = {'theta': 0.}; optimizer = Adam(.04); history = []
        for epoch in range(220):
            value, gradient = function.value_and_grad(parameters)
            history.append(dict(epoch=epoch, energy=value, theta=parameters['theta']))
            parameters = optimizer.step(parameters, gradient)
        energy = function(parameters); exact = problem.exact_energy()
        if abs(exact-reference) > 1e-8 or abs(energy-reference) > 1e-6:
            raise RuntimeError('H2 validation against PySCF FCI failed')
        rows.append(dict(bond_angstrom=float(distance), hartree_fock=problem.hartree_fock_energy,
            pyscf_fci=reference, nexqus_exact=exact, nexqus_vqe=energy,
            absolute_error_hartree=abs(energy-reference), nuclear_repulsion=problem.nuclear_repulsion,
            parameters=parameters, history=history,
            terms=[dict(coefficient=c, pauli=p) for c, p in problem.hamiltonian.terms]))
        print(f'R={distance:.3f} Å: HF={problem.hartree_fock_energy:.9f}, '
              f'FCI={reference:.9f}, VQE={energy:.9f}, error={abs(energy-reference):.2g} Ha')
        if detail is None or abs(distance-.735) < 1e-10: detail = (problem, circuit, parameters, rows[-1])
    problem, circuit, parameters, row = detail
    result = Simulator().run(circuit.bind(parameters), shots=0, retain_state=True)
    contributions = [dict(pauli=p, coefficient=c, expectation=result.expectation(p),
                          contribution=c*result.expectation(p)) for c, p in problem.hamiltonian.terms]
    explanation = circuit.explain(bindings=parameters, observable='ZZII',
        parameter_roles={'theta': 'weight'}, title='H₂ · mixing occupied and excited determinants')
    explanation.to_html(filename=output/'circuit.html')
    # Every prefix is inspectable, but only the final ansatz state is guaranteed
    # to lie in the two-electron sector. Prefix energies are not physical VQE iterates.
    trace = []
    for step, frame in enumerate(explanation.to_dict()['frames']):
        amplitudes = np.array(frame['amplitudes']); psi = amplitudes[:, 0]+1j*amplitudes[:, 1]
        p = abs(psi)**2
        trace.append(dict(step=step, energy_hartree=float(np.vdot(psi, problem.hamiltonian.dense()@psi).real),
            two_electron_probability=float(sum(p[i] for i in range(16) if bin(i).count('1') == 2))))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), layout='constrained')
    x = [r['bond_angstrom'] for r in rows]
    for name, label, color, style in [('hartree_fock', 'Classical RHF', '#0891b2', '--'),
            ('pyscf_fci', 'Classical FCI', '#1f2937', '-'), ('nexqus_vqe', 'NexQus VQE', '#7c3aed', ':')]:
        axes[0].plot(x, [r[name] for r in rows], style, marker='o', label=label, color=color)
    axes[0].set(title='H₂ / STO-3G · total energies', xlabel='Bond length / Å', ylabel='Energy / Hartree'); axes[0].legend()
    axes[1].plot([h['epoch'] for h in row['history']], [h['energy'] for h in row['history']], color='#7c3aed')
    axes[1].axhline(row['pyscf_fci'], color='#1f2937', linestyle='--', label='FCI')
    axes[1].set(title=f"VQE optimization · {row['bond_angstrom']} Å", xlabel='Update', ylabel='Total energy / Hartree'); axes[1].legend()
    axes[2].barh([c['pauli'] for c in contributions], [c['contribution'] for c in contributions], color='#7c3aed')
    axes[2].set(title='How measured terms build the energy', xlabel='coefficient × expectation / Hartree')
    fig.savefig(output/'comparison.png', dpi=160); fig.savefig(output/'comparison.svg'); plt.close(fig)
    payload = dict(sdk_version=nexqus.__version__, numpy=np.__version__, pyscf=pyscf.__version__,
        platform=platform.platform(), execution='local NumPy CPU double precision, exact expectations',
        source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in sorted(Path(nexqus.__file__).parent.glob('*.py'))},
        example_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        python=platform.python_version(), basis='sto-3g', mapping='Jordan–Wigner',
        mode_order='q0-first; spatial orbitals interleaved alpha/beta', n_qubits=4, n_electrons=2,
        energy_units='Hartree, includes nuclear repulsion', rows=rows, contributions=contributions,
        gate_prefix_trace=trace, prefix_warning='Only final ansatz states conserve two electrons; '
            'prefix energies are not a physical reaction path or optimization trajectory.',
        scope='Closed-shell H2 minimal-basis demonstration, not drug discovery or a quantum speedup benchmark.')
    (output/'metrics.json').write_text(json.dumps(payload, indent=2)+'\n', encoding='utf-8')
    print('Saved:', output.resolve()); return payload


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('.release-artifacts/h2'))
    generate(parser.parse_args().output)
