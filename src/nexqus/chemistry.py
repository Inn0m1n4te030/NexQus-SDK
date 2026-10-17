"""Optional PySCF RHF integrals -> NexQus Jordan–Wigner Pauli Hamiltonian.

Real, closed-shell, all-electron molecular orbitals only. No active-space,
relativistic, solvent, excited-state or drug-screening model is implied.
Units: input Angstrom, energies Hartree. q0-first interleaved alpha/beta modes.
"""
from dataclasses import dataclass
import itertools
import numbers
import numpy as np
from .circuit import Circuit, positive_integer
from .hamiltonian import PauliHamiltonian
from .parameters import real_scalar


# (left, right) -> (phase, product); identical nonidentity letters square to I.
_PRODUCT = {'XY': (1j, 'Z'), 'YZ': (1j, 'X'), 'ZX': (1j, 'Y'),
            'YX': (-1j, 'Z'), 'ZY': (-1j, 'X'), 'XZ': (-1j, 'Y')}


def _multiply(left, right):
    result = {}
    for a, ca in left.items():
        for b, cb in right.items():
            phase = 1; letters = []
            for x, y in zip(a, b):
                if x == 'I': z = y
                elif y == 'I': z = x
                elif x == y: z = 'I'
                else:
                    factor, z = _PRODUCT[x+y]; phase *= factor
                letters.append(z)
            word = ''.join(letters)
            result[word] = result.get(word, 0)+ca*cb*phase
    return result


def jordan_wigner(one_body, two_body, nuclear_repulsion=0., max_qubits=12, tolerance=1e-12):
    """Map real spatial MO integrals in chemists' (pq|rs) ordering.

    H = E_nuc + sum h[p,q] a†[p,s] a[q,s]
        + 1/2 sum (pq|rs) a†[p,s] a†[r,t] a[s,t] a[q,s].
    Modes 2*p and 2*p+1 are alpha and beta. No dense fermionic matrix is built.
    """
    if np.iscomplexobj(one_body) or np.iscomplexobj(two_body):
        raise ValueError('Real molecular integrals are required')
    h = np.asarray(one_body, float); eri = np.asarray(two_body, float)
    if h.ndim != 2 or h.shape[0] != h.shape[1] or not len(h):
        raise ValueError('one_body must be a nonempty square matrix')
    n = 2*len(h)
    if n > positive_integer(max_qubits, 'max_qubits'):
        raise ValueError('Molecule exceeds max_qubits; active-space reduction is not implemented')
    tolerance = real_scalar(tolerance)
    if tolerance < 0 or not np.isfinite(h).all() or eri.shape != (len(h),)*4 or not np.isfinite(eri).all():
        raise ValueError('Invalid molecular integrals or tolerance')
    if (not np.allclose(h, h.T, atol=1e-10, rtol=0)
            or not np.allclose(eri, eri.transpose(1, 0, 2, 3), atol=1e-10, rtol=0)
            or not np.allclose(eri, eri.transpose(2, 3, 0, 1), atol=1e-10, rtol=0)):
        raise ValueError('Integrals must have real chemists-order permutation symmetry')
    identity = 'I'*n
    terms = {identity: complex(real_scalar(nuclear_repulsion))}
    ladder = {}
    for mode in range(n):
        prefix = 'Z'*mode
        tail = 'I'*(n-mode-1)
        for creation in (False, True):
            ladder[mode, creation] = {prefix+'X'+tail: .5,
                                     prefix+'Y'+tail: -.5j if creation else .5j}
    def add(coefficient, operators):
        if coefficient == 0: return
        mapped = {identity: coefficient}
        for mode, creation in operators:
            mapped = _multiply(mapped, ladder[mode, creation])
        for word, value in mapped.items():
            terms[word] = terms.get(word, 0)+value
    for p, q in itertools.product(range(len(h)), repeat=2):
        for spin in (0, 1):
            add(h[p, q], [(2*p+spin, True), (2*q+spin, False)])
    for p, q, r, s in itertools.product(range(len(h)), repeat=4):
        for alpha, beta in itertools.product((0, 1), repeat=2):
            add(.5*eri[p, q, r, s], [(2*p+alpha, True), (2*r+beta, True),
                                     (2*s+beta, False), (2*q+alpha, False)])
    if any(abs(value.imag) > 1e-9 for value in terms.values()):
        raise ValueError('Mapped Hamiltonian is not Hermitian')
    return PauliHamiltonian(n, [(v.real, k) for k, v in terms.items() if abs(v.real) > tolerance])


@dataclass(frozen=True)
class MolecularProblem:
    hamiltonian: PauliHamiltonian
    n_electrons: int
    hartree_fock_energy: float
    nuclear_repulsion: float
    symbols: tuple
    coordinates_angstrom: tuple
    basis: str

    def __post_init__(self):
        if not isinstance(self.hamiltonian, PauliHamiltonian):
            raise TypeError('Expected PauliHamiltonian')
        if (self.hamiltonian.n_qubits % 2 or isinstance(self.n_electrons, bool)
                or not isinstance(self.n_electrons, numbers.Integral)
                or not 2 <= self.n_electrons <= self.hamiltonian.n_qubits or self.n_electrons % 2):
            raise ValueError('Expected closed-shell electron count in interleaved spin orbitals')

    def hartree_fock_circuit(self):
        circuit = Circuit(self.hamiltonian.n_qubits)
        for wire in range(self.n_electrons): circuit.x(wire)
        return circuit

    def molecule(self):
        """View the actual input geometry; energy metadata names its method."""
        from .molecules import Molecule, _ELEMENTS
        if any(s not in _ELEMENTS for s in self.symbols):
            raise ValueError('The geometry viewer does not support one of these elements')
        charge = sum(_ELEMENTS[s][0] for s in self.symbols)-self.n_electrons
        return Molecule(self.symbols, self.coordinates_angstrom, charge=charge,
                        name='Molecular problem', geometry_note='Fixed input geometry; '+self.basis+' basis')

    def exact_energy(self, max_qubits=10):
        """Dense reference restricted to the correct alpha/beta electron sector."""
        h = self.hamiltonian.dense(max_qubits=max_qubits)
        n = self.hamiltonian.n_qubits; occupied = self.n_electrons//2
        indices = [i for i in range(2**n)
                   if sum((i >> (n-1-q)) & 1 for q in range(0, n, 2)) == occupied
                   and sum((i >> (n-1-q)) & 1 for q in range(1, n, 2)) == occupied]
        return float(np.linalg.eigvalsh(h[np.ix_(indices, indices)])[0])


def molecular_hamiltonian(symbols, coordinates_angstrom, basis='sto-3g', charge=0, spin=0,
                          max_qubits=12):
    """Run local RHF and map all orbitals. Requires pip install .[chemistry].

    spin = N_alpha-N_beta must be zero in this first closed-shell driver.
    Qubit preflight happens before SCF and integral generation, after basis setup.
    No external services or quantum SDKs are used.
    """
    max_qubits = positive_integer(max_qubits, 'max_qubits')
    symbols = tuple(symbols)
    if not symbols or any(not isinstance(s, str) or not s.isalpha() for s in symbols):
        raise ValueError('Provide element symbols')
    if np.iscomplexobj(coordinates_angstrom): raise ValueError('Coordinates must be real')
    xyz = np.asarray(coordinates_angstrom, float)
    if xyz.shape != (len(symbols), 3) or not np.isfinite(xyz).all():
        raise ValueError('Coordinates must be finite with shape (atoms, 3), in Angstrom')
    if (isinstance(charge, bool) or not isinstance(charge, numbers.Integral)
            or isinstance(spin, bool) or not isinstance(spin, numbers.Integral) or spin != 0):
        raise ValueError('Provide integer charge and spin=0 (closed-shell only)')
    if not isinstance(basis, str) or not basis: raise ValueError('Provide a basis name')
    try:
        from pyscf import gto, scf, ao2mo
    except ImportError as exc:
        raise ImportError('Install nexqus-sdk[chemistry] for the PySCF driver') from exc
    mol = gto.M(atom=list(zip(symbols, xyz.tolist())), unit='Angstrom',
                basis=basis, charge=int(charge), spin=0, verbose=0)
    if 2*mol.nao_nr() > max_qubits:
        raise ValueError('Molecule exceeds max_qubits; active-space reduction is not implemented')
    if mol.nelectron < 2 or mol.nelectron > 2*mol.nao_nr():
        raise ValueError('Unsupported electron count')
    mean_field = scf.RHF(mol)
    mean_field.conv_tol = 1e-11
    energy = mean_field.kernel()
    if not mean_field.converged: raise RuntimeError('PySCF RHF did not converge')
    c = mean_field.mo_coeff
    h = c.T@mean_field.get_hcore()@c
    eri = ao2mo.kernel(mol, c, compact=False).reshape((c.shape[1],)*4)
    nuclear = float(mol.energy_nuc())
    return MolecularProblem(jordan_wigner(h, eri, nuclear, max_qubits=max_qubits),
        int(mol.nelectron), float(energy), nuclear, symbols,
        tuple(tuple(float(v) for v in row) for row in xyz), basis)


def h2_pair_ansatz(theta):
    """H2/STO-3G two-determinant family, not a general excitation operator.

    Output cos(theta/2)|1100> + sin(theta/2)|0011>. Intermediate decomposition
    states need not conserve particle number, but every final state does.
    """
    return Circuit(4).ry(theta, 0).cx(0, 1).cx(0, 2).cx(0, 3).x(0).x(1)


def h2_reference_problem():
    """Frozen H2/STO-3G problem at 0.735 Å; no PySCF installation required.

    PySCF 2.14.0 RHF integrals, neutral singlet, interleaved alpha/beta modes.
    Nuclear repulsion is included. Regenerate with molecular_hamiltonian();
    coefficients and FCI validation are recorded in the 2026-10-17 report.
    This fixture cannot be used at other bond lengths by moving its geometry.
    """
    terms = [(-.09057898608834797, 'IIII'), (-.22575349222402394, 'IIIZ'),
        (-.22575349222402394, 'IIZI'), (.17464343068300442, 'IIZZ'),
        (.1721839326191555, 'IZII'), (.1209126326177663, 'IZIZ'),
        (.16614543256382414, 'IZZI'), (-.04523279994605785, 'XXYY'),
        (.04523279994605785, 'XYYX'), (.04523279994605785, 'YXXY'),
        (-.04523279994605785, 'YYXX'), (.1721839326191555, 'ZIII'),
        (.16614543256382414, 'ZIIZ'), (.1209126326177663, 'ZIZI'),
        (.16892753870087907, 'ZZII')]
    return MolecularProblem(PauliHamiltonian(4, terms), 2, -1.116998996754004,
        .7199689944489797, ('H', 'H'), ((0., 0., 0.), (0., 0., .735)), 'sto-3g')
