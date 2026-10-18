"""Real weighted Pauli sums, circuit expectations and exact VQE gradients."""
import numpy as np
from .circuit import Circuit, I, X, Y, Z, positive_integer
from .gradients import QuantumFunction
from .parameters import real_scalar
from .simulator import resolve_simulator


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
        return float(sum(c*result.expectation_native(p) for c, p in self.terms))

    def dense(self, max_qubits=10, *, simulator=None,backend=None,device=None,precision=None):
        if self.n_qubits > positive_integer(max_qubits, 'max_qubits'):
            raise ValueError('Dense Hamiltonian exceeds max_qubits')
        b=resolve_simulator(simulator,backend=backend,device=device,precision=precision).backend
        matrices = {k:b.array(v) for k,v in dict(I=I,X=X,Y=Y,Z=Z).items()}
        h = b.zeros((2**self.n_qubits,)*2)
        for coefficient, word in self.terms:
            matrix = b.array([[1]])
            for letter in word:
                matrix = b.torch.kron(matrix.contiguous(),matrices[letter]) if b.torch else np.kron(matrix,matrices[letter])
            h += coefficient*matrix
        return h

    def function(self, template, simulator=None, **options):
        return HamiltonianFunction(template,self,simulator,**options)


class HamiltonianFunction(QuantumFunction):
    """One evolved state per value/shift; weighted Pauli reduction on its device."""
    def __init__(self, template, hamiltonian,simulator=None,*,backend=None,device=None,precision=None,noise=None):
        if not isinstance(template, Circuit) or not isinstance(hamiltonian, PauliHamiltonian):
            raise TypeError('Expected Circuit and PauliHamiltonian')
        if template.n_qubits != hamiltonian.n_qubits:
            raise ValueError('Hamiltonian/circuit qubit count mismatch')
        if template.dynamic:
            raise ValueError('HamiltonianFunction requires a static circuit')
        self.hamiltonian=hamiltonian
        super().__init__(template,'I'*template.n_qubits,simulator,noise,
                         backend=backend,device=device,precision=precision)

    def _evaluate(self,circuit):
        if circuit.dynamic: raise ValueError('HamiltonianFunction requires a static circuit')
        return self.hamiltonian.expectation(self.simulator.run(circuit,noise=self.noise,
            shots=0,retain_state=True,native_result=True))
