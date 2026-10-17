"""Native Windows molecular views, including a local H2 VQE result.

python examples/molecular_viewer.py
Open .release-artifacts/molecules/gallery.html or h2_vqe.html in your browser.
No PySCF, RDKit, cloud, QPU, account or web viewer dependency is required.
"""
import argparse
import json
from pathlib import Path
from nexqus import molecular_gallery, molecule, Parameter, Adam
from nexqus.chemistry import h2_reference_problem, h2_pair_ansatz


def generate(output):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    gallery = molecular_gallery()
    gallery.to_html(output/'gallery.html')
    for name in ('H2', 'CH4', 'H2O', 'C2H4', 'C6H6'):
        structure = molecule(name)
        structure.draw(output/(name+'.svg'))
        structure.to_xyz(output/(name+'.xyz'))
    problem = h2_reference_problem()
    energy = problem.hamiltonian.function(h2_pair_ansatz(Parameter('theta')))
    parameters = {'theta': 0.}; optimizer = Adam(.04)
    for _ in range(180):
        _, gradient = energy.value_and_grad(parameters)
        parameters = optimizer.step(parameters, gradient)
    final = energy(parameters); reference = problem.exact_energy()
    if abs(final-reference) > 2e-7: raise RuntimeError('H2 VQE failed its reference check')
    report = problem.molecule().view(energies_hartree={
        'Classical Hartree–Fock': problem.hartree_fock_energy,
        'Classical exact / FCI reference': reference, 'NexQus VQE': final},
        calculation_note='H2 at fixed 0.735 Å · STO-3G · two electrons / four spin-orbital qubits. '
        'VQE calculated locally using the frozen PySCF-derived Hamiltonian; total energies include nuclear repulsion.')
    report.to_html(output/'h2_vqe.html')
    (output/'h2_vqe.json').write_text(json.dumps(report.to_dict(), indent=2)+'\n', encoding='utf-8')
    print('Gallery:', (output/'gallery.html').resolve())
    print('H2 molecule with VQE results:', (output/'h2_vqe.html').resolve())
    print(f'H2 energy: {final:.12f} Ha; absolute reference error: {abs(final-reference):.3g} Ha')
    print('CH4 / H2O / C2H4 / C6H6 gallery entries are labelled idealized geometry examples.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('.release-artifacts/molecules'))
    generate(parser.parse_args().output)
