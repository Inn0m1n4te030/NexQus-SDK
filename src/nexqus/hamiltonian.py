"""Real weighted Pauli sums, circuit expectations and exact VQE gradients."""
import numpy as np
from .circuit import Circuit, I, X, Y, Z, positive_integer
from .gradients import QuantumFunction, _names
from .parameters import real_scalar


class PauliHamiltonian:
    """Hermitian Pauli sum in q0-first order, including constant I terms.

    Terms are (real coefficient, Pauli string). Duplicate strings combine.
    dense() is an explicitly bounded reference tool, not the execution path.
    """
    def __init__(self, n_qubits, terms):
        self.n_qubits = positive_integer(n_qubits, 'n_qubits')
        combined = {}
        for coefficient, word in terms:
            coefficient = real_scalar(coefficient)
            if not isinstance(word, str) or len(word) != self.n_qubits or set(word)-set('IXYZ'):
                raise ValueError('Provide a q0-first Pauli string per term')
            combined[word] = real_scalar(combined.get(word, 0.)+coefficient)
        self.terms = tuple((v, k) for k, v in sorted(combined.items()) if v != 0)

    def expectation(self, result):
        if result.metadata['n_qubits'] != self.n_qubits:
            raise ValueError('Hamiltonian/result qubit count mismatch')
        return float(sum(c*result.expectation(p) for c, p in self.terms))

    def dense(self, max_qubits=10):
        if self.n_qubits > positive_integer(max_qubits, 'max_qubits'):
            raise ValueError('Dense Hamiltonian exceeds max_qubits')
        matrices = dict(I=I, X=X, Y=Y, Z=Z)
        h = np.zeros((2**self.n_qubits,)*2, complex)
        for coefficient, word in self.terms:
            matrix = np.array([[1]], complex)
            for letter in word:
                matrix = np.kron(matrix, matrices[letter])
            h += coefficient*matrix
        return h

    def function(self, template):
        return HamiltonianFunction(template, self)


class HamiltonianFunction:
    """Weighted expectation and analytic gradient; one simulation per term.

    This first implementation favors inspectability over measurement grouping.
    """
    def __init__(self, template, hamiltonian):
        if not isinstance(template, Circuit) or not isinstance(hamiltonian, PauliHamiltonian):
            raise TypeError('Expected Circuit and PauliHamiltonian')
        if template.n_qubits != hamiltonian.n_qubits:
            raise ValueError('Hamiltonian/circuit qubit count mismatch')
        if template.dynamic:
            raise ValueError('HamiltonianFunction requires a static circuit')
        self.template = template
        self.terms = [(c, QuantumFunction(template, word)) for c, word in hamiltonian.terms]

    def __call__(self, bindings):
        self.template.bind(bindings)
        return float(sum(c*f(bindings) for c, f in self.terms))

    def value_and_grad(self, bindings, wrt=None):
        self.template.bind(bindings)
        names = _names(wrt, self.template.parameters)
        value = 0.; gradient = dict.fromkeys(names, 0.)
        for coefficient, function in self.terms:
            v, g = function.value_and_grad(bindings, wrt=names)
            value += coefficient*v
            for name in names:
                gradient[name] += coefficient*g[name]
        return float(value), gradient

    def gradient(self, bindings, wrt=None):
        return self.value_and_grad(bindings, wrt)[1]
