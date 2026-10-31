"""NexQus: own circuit engine on NumPy or PyTorch tensors."""
from .circuit import Circuit, circuit
from .noise import Noise, NoiseModel
from .simulator import Simulator, Result, CompiledCircuit
from .parameters import Parameter
from .gradients import QuantumFunction
from .optimizers import Adam, SGD
from .learning import QuantumRegressor, QuantumBinaryClassifier
from .generative import QuantumDistribution, QGAN, distribution_metrics
from .hamiltonian import PauliHamiltonian, HamiltonianFunction
from .molecules import Molecule, MolecularView, molecule, molecular_gallery
from . import algorithms, templates, nlp

__version__ = "1.4.0"
__all__ = ["Circuit", "circuit", "Parameter", "Noise", "NoiseModel", "Simulator", "Result", "CompiledCircuit", "QuantumFunction", "QuantumRegressor", "QuantumBinaryClassifier", "Adam", "SGD", "algorithms", "templates"]
__all__ += ["QuantumDistribution", "QGAN", "distribution_metrics", "PauliHamiltonian", "HamiltonianFunction"]
__all__ += ["Molecule", "MolecularView", "molecule", "molecular_gallery"]
__all__ += ["nlp"]
