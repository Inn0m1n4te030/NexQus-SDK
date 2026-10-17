"""Molecular geometry and offline viewers; no chemistry driver is required.

Coordinates are Angstrom. Bonds are supplied connectivity or a labelled distance
heuristic. A geometry visualization does not calculate electronic structure.
"""
from collections import Counter
import copy
import html
import math
import numbers
from pathlib import Path
import numpy as np
from .parameters import real_scalar


# Atomic number, display color, covalent radius (Angstrom). Display radii are
# illustrative; automatic connectivity is only a distance heuristic.
_ELEMENTS = {
    'H': (1, '#e9edf5', .31), 'He': (2, '#8de2ec', .28),
    'Li': (3, '#b18cda', 1.28), 'Be': (4, '#9bcb6a', .96),
    'B': (5, '#d6a082', .84), 'C': (6, '#536079', .76),
    'N': (7, '#4268d6', .71), 'O': (8, '#df5264', .66),
    'F': (9, '#68bd82', .57), 'Ne': (10, '#83d6df', .58),
    'Na': (11, '#a384d0', 1.66), 'Mg': (12, '#8bc474', 1.41),
    'Al': (13, '#a8aeba', 1.21), 'Si': (14, '#c8a78b', 1.11),
    'P': (15, '#ec9b47', 1.07), 'S': (16, '#e1c35a', 1.05),
    'Cl': (17, '#56b67b', 1.02), 'Ar': (18, '#8ac7df', 1.06),
    'Br': (35, '#a75f50', 1.20), 'I': (53, '#9473b7', 1.39),
}


class Molecule:
    """A validated, bounded geometry; use explicit bonds for chemical accuracy.

    bonds = [(atom_index, atom_index, order)] with zero-based atom indices and
    order 1, 1.5, 2 or 3. None infers single bonds by a distance heuristic;
    [] deliberately supplies no bonds. A maximum of 256 atoms is supported.
    """
    def __init__(self, symbols, coordinates_angstrom, bonds=None, *, name='Custom molecule',
                 charge=0, geometry_note='Caller-supplied fixed geometry'):
        symbols = tuple(symbols)
        if not 1 <= len(symbols) <= 256 or any(not isinstance(s, str) or s not in _ELEMENTS for s in symbols):
            raise ValueError('Provide 1..256 supported element symbols')
        if not isinstance(name, str) or not name.strip() or len(name) > 512:
            raise ValueError('Provide a molecule name of 1..512 characters')
        if not isinstance(geometry_note, str) or len(geometry_note) > 2048:
            raise ValueError('geometry_note must be a string of at most 2048 characters')
        if isinstance(charge, bool) or not isinstance(charge, numbers.Integral):
            raise ValueError('charge must be an integer')
        electrons = sum(_ELEMENTS[s][0] for s in symbols)-int(charge)
        if electrons < 0: raise ValueError('Charge exceeds the available electrons')
        if np.iscomplexobj(coordinates_angstrom): raise ValueError('Coordinates must be real')
        xyz = np.array(coordinates_angstrom, dtype=float, copy=True)
        if xyz.shape != (len(symbols), 3) or not np.isfinite(xyz).all() or np.any(abs(xyz) > 1e6):
            raise ValueError('Provide finite (atoms, 3) coordinates in Angstrom, bounded to ±1e6')
        for i in range(len(xyz)):
            if any(np.linalg.norm(xyz[i]-xyz[j]) < 1e-8 for j in range(i)):
                raise ValueError('Two atoms cannot occupy the same position')
        inferred = bonds is None
        if inferred:
            bonds = [(i, j, 1) for i in range(len(xyz)) for j in range(i)
                     if np.linalg.norm(xyz[i]-xyz[j]) <= 1.25*(_ELEMENTS[symbols[i]][2]+_ELEMENTS[symbols[j]][2])]
        checked = []; seen = set()
        for bond in bonds:
            if len(bond) != 3: raise ValueError('Each bond needs two atom indices and an order')
            a, b, order = bond
            if any(isinstance(v, bool) or not isinstance(v, numbers.Integral)
                   or not 0 <= v < len(symbols) for v in (a, b)) or a == b:
                raise ValueError('Bond indices must be distinct, zero-based atom indices')
            order = real_scalar(order)
            if order not in (1, 1.5, 2, 3): raise ValueError('Bond order must be 1, 1.5, 2 or 3')
            pair = tuple(sorted((int(a), int(b))))
            if pair in seen: raise ValueError('Duplicate bond')
            seen.add(pair); checked.append((*pair, order))
        xyz.flags.writeable = False
        self.symbols = symbols; self.coordinates_angstrom = xyz
        self.bonds = tuple(sorted(checked)); self.name = name; self.charge = int(charge)
        self.geometry_note = geometry_note
        self.bond_source = 'Distance heuristic; verify connectivity and bond orders' if inferred else 'Explicit connectivity'

    @property
    def formula(self):
        counts = Counter(self.symbols)
        order = (['C']+(['H'] if 'H' in counts else [])+sorted(set(counts)-{'C', 'H'})
                 if 'C' in counts else sorted(counts))
        return ''.join(s+(str(counts[s]) if counts[s] > 1 else '') for s in order)

    @property
    def n_electrons(self):
        return sum(_ELEMENTS[s][0] for s in self.symbols)-self.charge

    def _index(self, index):
        if isinstance(index, bool) or not isinstance(index, numbers.Integral) or not 0 <= index < len(self.symbols):
            raise ValueError('Provide an atom index in range')
        return int(index)

    def distance(self, a, b):
        a, b = self._index(a), self._index(b)
        return float(np.linalg.norm(self.coordinates_angstrom[a]-self.coordinates_angstrom[b]))

    def angle(self, a, center, b):
        a, center, b = map(self._index, (a, center, b))
        if len({a, center, b}) != 3: raise ValueError('An angle needs three distinct atoms')
        u = self.coordinates_angstrom[a]-self.coordinates_angstrom[center]
        v = self.coordinates_angstrom[b]-self.coordinates_angstrom[center]
        return float(np.degrees(np.arccos(np.clip(u@v/(np.linalg.norm(u)*np.linalg.norm(v)), -1, 1))))

    def to_dict(self):
        # STO-3G mode estimate is limited to H..Ne, independent of connectivity.
        estimate = 2*sum(1 if _ELEMENTS[s][0] <= 2 else 5 for s in self.symbols) if all(_ELEMENTS[s][0] <= 10 for s in self.symbols) else None
        return dict(name=self.name, formula=self.formula, charge=self.charge,
            n_electrons=self.n_electrons, geometry_note=self.geometry_note, bond_source=self.bond_source,
            units='Angstrom', sto3g_qubits_estimate=estimate,
            atoms=[dict(index=i, symbol=s, label=f'{s}{i+1}', position=xyz.tolist(),
                        color=_ELEMENTS[s][1], display_radius=.22+.15*math.sqrt(_ELEMENTS[s][2]))
                   for i, (s, xyz) in enumerate(zip(self.symbols, self.coordinates_angstrom))],
            bonds=[dict(a=a, b=b, order=o, length=self.distance(a, b)) for a, b, o in self.bonds])

    def to_xyz(self, filename=None):
        comment = self.name.replace('\n', ' ').replace('\r', ' ')+' | Angstrom | '+self.geometry_note.replace('\n', ' ').replace('\r', ' ')
        text = f'{len(self.symbols)}\n{comment}\n'+''.join(
            f'{s} {v[0]:.10f} {v[1]:.10f} {v[2]:.10f}\n' for s, v in zip(self.symbols, self.coordinates_angstrom))
        if filename is not None: Path(filename).write_text(text, encoding='utf-8')
        return text

    def draw(self, filename=None):
        """Static SVG overview; view() supplies interactive 3D rotation."""
        from .molecule_html import static_svg
        svg = static_svg(self.to_dict())
        if filename is not None: Path(filename).write_text(svg, encoding='utf-8')
        return svg

    def view(self, *, energies_hartree=None, calculation_note=None):
        return MolecularView([self], energies_hartree=energies_hartree, calculation_note=calculation_note)

    def _repr_html_(self):
        return self.view()._repr_html_()


class MolecularView:
    """Self-contained, offline SVG viewer with geometry and result provenance."""
    def __init__(self, molecules, *, energies_hartree=None, calculation_note=None):
        molecules = tuple(molecules)
        if not 1 <= len(molecules) <= 32 or any(not isinstance(m, Molecule) for m in molecules):
            raise ValueError('Provide 1..32 Molecule objects')
        data = [m.to_dict() for m in molecules]
        if energies_hartree is not None:
            if len(data) != 1 or not isinstance(energies_hartree, dict) or not 1 <= len(energies_hartree) <= 32:
                raise ValueError('Attach 1..32 labelled energies to a single molecule view')
            if not isinstance(calculation_note, str) or not calculation_note.strip() or len(calculation_note) > 2048:
                raise ValueError('Describe the method, basis and geometry in calculation_note')
            values = {}
            for label, value in energies_hartree.items():
                if not isinstance(label, str) or not label or len(label) > 160: raise ValueError('Invalid energy label')
                values[label] = real_scalar(value)
            data[0]['energies_hartree'] = values; data[0]['calculation_note'] = calculation_note
        self._data = dict(schema_version=1, molecules=data)

    def to_dict(self):
        return copy.deepcopy(self._data)

    def to_html(self, filename=None):
        from .molecule_html import render_molecules
        report = MoleculeHTML(render_molecules(self._data))
        if filename is not None: Path(filename).write_text(report, encoding='utf-8')
        return report

    def _repr_html_(self):
        return self.to_html()._repr_html_()


class MoleculeHTML(str):
    def _repr_html_(self):
        return ('<iframe title="NexQus molecular viewer" sandbox="allow-scripts allow-downloads" '
                'style="width:100%;height:940px;border:0" srcdoc="'+html.escape(self, quote=True)+'"></iframe>')


def molecule(name):
    """Example fixed geometries: H2, CH4, H2O, C2H4 and C6H6.

    Except H2's reference geometry, coordinates are idealized illustrative
    structures, not calculated/optimized geometries or electronic densities.
    """
    note = 'Idealized illustrative geometry; no geometry optimization or energy calculation attached'
    if name == 'H2':
        return Molecule(['H', 'H'], [[0, 0, -.3675], [0, 0, .3675]], [(0, 1, 1)],
                        name='Hydrogen', geometry_note='Fixed H–H distance 0.735 Å; matches the H2/STO-3G tutorial')
    if name == 'CH4':
        a = 1.09/np.sqrt(3)
        return Molecule(['C']+['H']*4, [[0, 0, 0], [a, a, a], [a, -a, -a], [-a, a, -a], [-a, -a, a]],
                        [(0, i, 1) for i in range(1, 5)], name='Methane', geometry_note=note+'; C–H 1.09 Å, tetrahedral')
    if name == 'H2O':
        a = np.deg2rad(104.5/2)
        return Molecule(['O', 'H', 'H'], [[0, 0, 0], [.958*np.sin(a), .958*np.cos(a), 0],
                        [-.958*np.sin(a), .958*np.cos(a), 0]], [(0, 1, 1), (0, 2, 1)], name='Water', geometry_note=note)
    if name == 'C2H4':
        x = .6695; dx = 1.087*np.cos(np.deg2rad(58.7)); dy = 1.087*np.sin(np.deg2rad(58.7))
        return Molecule(['C', 'C']+['H']*4, [[-x, 0, 0], [x, 0, 0], [-x-dx, dy, 0], [-x-dx, -dy, 0],
                        [x+dx, dy, 0], [x+dx, -dy, 0]], [(0, 1, 2), (0, 2, 1), (0, 3, 1), (1, 4, 1), (1, 5, 1)],
                        name='Ethene', geometry_note=note+'; planar C=C double bond')
    if name == 'C6H6':
        points = [[r*np.cos(t), r*np.sin(t), 0] for r in (1.397, 2.487) for t in np.arange(6)*np.pi/3]
        return Molecule(['C']*6+['H']*6, points, [(i, (i+1)%6, 1.5) for i in range(6)]+[(i, i+6, 1) for i in range(6)],
                        name='Benzene', geometry_note=note+'; aromatic bonds shown with order 1.5')
    raise ValueError('Choose H2, CH4, H2O, C2H4 or C6H6, or construct a custom Molecule')


def molecular_gallery():
    """Native Windows-compatible gallery; no PySCF, RDKit or network required."""
    return MolecularView([molecule(name) for name in ('H2', 'CH4', 'H2O', 'C2H4', 'C6H6')])
