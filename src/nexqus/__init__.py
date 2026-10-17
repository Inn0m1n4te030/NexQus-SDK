"""NexQus: own circuit engine on NumPy or PyTorch tensors."""
from .circuit import Circuit, circuit
from .noise import Noise, NoiseModel
from .simulator import Simulator, Result, CompiledCircuit
from .parameters import Parameter
from .gradients import QuantumFunction
from .optimizers import Adam, SGD
from .learning import QuantumRegressor, QuantumBinaryClassifier
from . import algorithms, templates

__version__ = "1.2.0"
__all__ = ["Circuit", "circuit", "Parameter", "Noise", "NoiseModel", "Simulator", "Result", "CompiledCircuit", "QuantumFunction", "QuantumRegressor", "QuantumBinaryClassifier", "Adam", "SGD", "algorithms", "templates"]
