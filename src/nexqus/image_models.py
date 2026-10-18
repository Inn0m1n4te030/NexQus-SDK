"""Optional CPU/CUDA PyTorch image generators backed by NexQus circuits.

Quantum states, batched angles and gradients remain native Torch tensors.
"""
import math
import numbers
import numpy as np
try:
    import torch
    from torch import nn
except ImportError as exc:
    raise ImportError('Install the source checkout with .[image-gan] to use image generators') from exc
from .circuit import Circuit, positive_integer
from .parameters import Parameter, real_scalar
from .generative import QuantumDistribution
from .torch_circuit import TorchCircuit,tensor_device


def _bounded_int(value, name, maximum):
    value = positive_integer(value, name)
    if value > maximum: raise ValueError(f'{name} must be at most {maximum}')
    return value


def _seed(value):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or not 0 <= value <= 2**63-1024:
        raise ValueError('seed must be an integer between 0 and 2**63-1024')
    return int(value)


def _image_shape(image_size, channels):
    image_size = _bounded_int(image_size, 'image_size', 64)
    if image_size < 4: raise ValueError('image_size must be at least 4')
    channels = positive_integer(channels, 'channels')
    if channels not in (1, 3): raise ValueError('channels must be 1 or 3')
    return int(image_size), int(channels)


def _architecture(value, image_size):
    if value not in ('mlp', 'conv'): raise ValueError('architecture must be mlp or conv')
    if value == 'conv' and image_size & (image_size-1):
        raise ValueError('Convolutional image sizes must be powers of two (4..64)')
    return value


def _conv_decoder(features, image_size, channels, hidden):
    """Project to a spatial grid, then learn local features at each resolution."""
    width = max(16, hidden)
    layers = [nn.Linear(features, width*4*4), nn.Unflatten(1, (width, 4, 4)),
              nn.BatchNorm2d(width), nn.ReLU()]
    resolution = 4
    while resolution < image_size:
        output = max(16, width//2)
        layers += [nn.Upsample(scale_factor=2, mode='nearest'),
                   nn.Conv2d(width, output, 3, padding=1, bias=False),
                   nn.BatchNorm2d(output), nn.ReLU()]
        width = output; resolution *= 2
    layers += [nn.Conv2d(width, channels, 3, padding=1), nn.Sigmoid()]
    return nn.Sequential(*layers)


class _Probabilities:
    @staticmethod
    def apply(weights,angles,layer):
        bindings={name:weights[i] for i,name in enumerate(layer.weight_names)}
        bindings.update({name:angles[:,i] for i,name in enumerate(layer.input_names)})
        return layer.tensor_circuit(bindings)


class QuantumProbabilityLayer(nn.Module):
    """Bounded batched Torch CPU/CUDA evolution with native autograd.

    Input shape (batch, n_qubits), output (batch, 2**n_qubits). Every input
    angle and circuit weight has analytic first derivatives. Batch cap: 128.
    """
    def __init__(self, n_qubits=4, depth=1, seed=7):
        super().__init__()
        seed = _seed(seed)
        self.n_qubits = _bounded_int(n_qubits, 'n_qubits', 8)
        self.depth = _bounded_int(depth, 'depth', 6)
        self.input_names = [f'z{i}' for i in range(self.n_qubits)]
        circuit = Circuit(self.n_qubits)
        for q, name in enumerate(self.input_names): circuit.ry(Parameter(name), q)
        self.weight_names = []
        for _ in range(self.depth):
            for q in range(self.n_qubits):
                for gate in ('ry', 'rz', 'rx'):
                    name = f'w{len(self.weight_names)}'; self.weight_names.append(name)
                    getattr(circuit, gate)(Parameter(name), q)
            for q in range(self.n_qubits-1): circuit.cx(q, q+1)
            if self.n_qubits > 2: circuit.cx(self.n_qubits-1, 0)
        self.function = QuantumDistribution(circuit, max_qubits=8)
        self.tensor_circuit=TorchCircuit(circuit,max_qubits=8,precision='single')
        rng = torch.Generator().manual_seed(seed)
        self.weights = nn.Parameter(.4*torch.randn(len(self.weight_names), generator=rng))

    def bindings(self, weights, angles):
        return dict(zip(self.weight_names+self.input_names,
                        [float(v) for v in weights]+[float(v) for v in angles]))

    def forward(self, angles):
        if (not isinstance(angles, torch.Tensor) or angles.device!=self.weights.device
                or self.weights.device.type not in ('cpu','cuda') or angles.dtype != self.weights.dtype
                or angles.dtype not in (torch.float32, torch.float64)):
            raise ValueError('Quantum image layers require CPU/CUDA float32/float64 tensors matching their weights')
        if angles.ndim != 2 or angles.shape[1] != self.n_qubits or not 1 <= len(angles) <= 128:
            raise ValueError('Angles must have shape (batch=1..128, n_qubits)')
        if not torch.isfinite(angles).all() or not torch.isfinite(self.weights).all():
            raise ValueError('Quantum angles and weights must be finite')
        return _Probabilities.apply(self.weights, angles, self)

    def explain(self, angles):
        """Circuit-state explanation at fixed input angles and learned weights."""
        bindings = self.bindings(self.weights.detach().cpu().numpy(),
                                 angles.detach().cpu().numpy() if torch.is_tensor(angles) else np.asarray(angles))
        return self.function.template.explain(bindings=bindings,
            backend='torch',device=str(self.weights.device),
            parameter_roles={**{n: 'feature' for n in self.input_names},
                             **{n: 'weight' for n in self.weight_names}},
            title='Image generator · latent angles through the quantum circuit')


class _Generator(nn.Module):
    def _setup(self, image_size, channels, latent_dim, num_classes):
        self.image_size, self.channels = _image_shape(image_size, channels)
        self.latent_dim = _bounded_int(latent_dim, 'latent_dim', 64)
        self.num_classes = _bounded_int(num_classes, 'num_classes', 32)

    def _inputs(self, noise, labels):
        reference = next(self.parameters())
        if (not isinstance(noise, torch.Tensor) or noise.device!=reference.device
                or noise.dtype != reference.dtype or reference.device.type not in ('cpu','cuda')
                or noise.ndim != 2 or noise.shape[1] != self.latent_dim
                or not 1 <= len(noise) <= 128 or not torch.isfinite(noise).all()):
            raise ValueError('Provide finite CPU/CUDA noise with shape (batch=1..128, latent_dim) matching model device/dtype')
        if (not isinstance(labels, torch.Tensor) or labels.device!=reference.device
                or labels.dtype != torch.long or labels.shape != (len(noise),)
                or torch.any(labels < 0) or torch.any(labels >= self.num_classes)):
            raise ValueError('Provide int64 class labels matching model device, one valid label per noise vector')

    def quantum_layers(self):
        return [m for m in self.modules() if isinstance(m, QuantumProbabilityLayer)]

    def parameter_effect(self, noise, labels, index=0, delta=.2):
        """Image counterfactuals for one quantum weight, at fixed noise/class.

        Returns images at w-delta, w, w+delta and a central finite-difference
        pixel sensitivity. This is a local parameter experiment, not attribution.
        Call between training steps; weights and train/eval mode are restored.
        """
        self._inputs(noise, labels)
        if len(noise) != 1: raise ValueError('Explain one fixed sample at a time')
        if isinstance(index, bool) or not isinstance(index, int) or index < 0: raise ValueError('Invalid quantum weight index')
        delta = real_scalar(delta)
        if delta <= 0: raise ValueError('delta must be positive')
        layers = self.quantum_layers()
        target = None; flat = index
        for circuit_index, layer in enumerate(layers):
            if flat < layer.weights.numel(): target = layer; break
            flat -= layer.weights.numel()
        if target is None: raise ValueError('Quantum weight index out of range')
        previous = self.training; original = target.weights[flat].detach().clone()
        self.eval()
        try:
            with torch.no_grad():
                target.weights[flat].copy_(original-delta); minus = self(noise, labels).clone()
                target.weights[flat].copy_(original); center = self(noise, labels).clone()
                target.weights[flat].copy_(original+delta); plus = self(noise, labels).clone()
        finally:
            with torch.no_grad(): target.weights[flat].copy_(original)
            self.train(previous)
        return dict(parameter=f'quantum circuit {circuit_index} / {target.weight_names[flat]}',
                    value=float(original), delta=delta, minus=minus, center=center, plus=plus,
                    sensitivity=(plus-minus)/(2*delta))


class QuantumPatchGenerator(_Generator):
    """Conditional quantum-generated pixel patches with classical angle encoding.

    Each channel/patch has its own circuit. The last qubit is an ancilla;
    conditional ancilla=0 probabilities are scaled by their maximum into [0,1].
    Every nonzero patch therefore has a brightest pixel of 1. No image decoder
    is used. Pixel order is channel, patch row, patch column, row-major pixels.
    """
    def __init__(self, image_size=8, channels=1, latent_dim=4, num_classes=2,
                 patch_size=4, depth=1, seed=7):
        super().__init__(); self._setup(image_size, channels, latent_dim, num_classes)
        seed = _seed(seed); depth = _bounded_int(depth, 'depth', 6)
        image_size, channels, latent_dim, num_classes = self.image_size, self.channels, self.latent_dim, self.num_classes
        patch_size = positive_integer(patch_size, 'patch_size')
        if patch_size not in (2, 4, 8): raise ValueError('patch_size must be 2, 4 or 8')
        if self.image_size % patch_size: raise ValueError('image_size must be divisible by patch_size')
        self.patch_size = patch_size; self.n_qubits = int(math.log2(patch_size**2))+1
        count = channels*(self.image_size//patch_size)**2
        if count > 64: raise ValueError('Patch generator is limited to 64 circuits per image; use larger patches or smaller images')
        self.config = dict(kind='patch', image_size=image_size, channels=channels, latent_dim=latent_dim,
                           num_classes=num_classes, patch_size=patch_size, depth=depth, seed=seed)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.encoder = nn.Linear(latent_dim, self.n_qubits)
            self.condition = nn.Embedding(num_classes, self.n_qubits)
        self.circuits = nn.ModuleList([QuantumProbabilityLayer(self.n_qubits, depth, seed+i) for i in range(count)])

    def encoded_angles(self, noise, labels):
        return math.pi*torch.tanh(self.encoder(noise)+self.condition(labels))

    def forward(self, noise, labels):
        self._inputs(noise, labels); angles = self.encoded_angles(noise, labels)
        patches = []
        for circuit in self.circuits:
            joint = circuit(angles).reshape(len(noise), self.patch_size**2, 2)
            selected = joint[:, :, 0]
            conditional = selected/selected.sum(dim=1, keepdim=True).clamp_min(1e-12)
            patches.append(conditional/conditional.amax(dim=1, keepdim=True).clamp_min(1e-12))
        side = self.image_size//self.patch_size
        packed = torch.stack(patches, dim=1).reshape(len(noise), self.channels, side, side, self.patch_size, self.patch_size)
        return packed.permute(0, 1, 2, 4, 3, 5).reshape(len(noise), self.channels, self.image_size, self.image_size)

    def circuit_report(self, noise, labels):
        self._inputs(noise, labels)
        return self.circuits[0].explain(self.encoded_angles(noise[:1], labels[:1])[0])


class HybridImageGenerator(_Generator):
    """Conditional quantum Z features -> MLP or convolutional image decoder."""
    def __init__(self, image_size=8, channels=1, latent_dim=4, num_classes=2,
                 n_qubits=4, depth=1, hidden=64, seed=7, architecture='mlp'):
        super().__init__(); self._setup(image_size, channels, latent_dim, num_classes)
        seed = _seed(seed)
        image_size, channels, latent_dim, num_classes = self.image_size, self.channels, self.latent_dim, self.num_classes
        hidden = _bounded_int(hidden, 'hidden', 512)
        architecture = _architecture(architecture, image_size)
        self.quantum = QuantumProbabilityLayer(n_qubits, depth, seed)
        n_qubits, depth = self.quantum.n_qubits, self.quantum.depth
        self.config = dict(kind='hybrid', image_size=image_size, channels=channels, latent_dim=latent_dim,
                           num_classes=num_classes, n_qubits=n_qubits, depth=depth, hidden=hidden, seed=seed,
                           architecture=architecture)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.encoder = nn.Linear(latent_dim, n_qubits); self.condition = nn.Embedding(num_classes, n_qubits)
            self.decoder = (_conv_decoder(n_qubits, image_size, channels, hidden) if architecture == 'conv'
                else nn.Sequential(nn.Linear(n_qubits, hidden), nn.LeakyReLU(.2),
                    nn.Linear(hidden, channels*image_size**2), nn.Sigmoid()))
        signs = [[1-2*((basis >> (n_qubits-1-wire)) & 1) for wire in range(n_qubits)] for basis in range(2**n_qubits)]
        self.register_buffer('z_signs', torch.tensor(signs, dtype=torch.float32))

    def encoded_angles(self, noise, labels):
        return math.pi*torch.tanh(self.encoder(noise)+self.condition(labels))

    def forward(self, noise, labels):
        self._inputs(noise, labels)
        features = self.quantum(self.encoded_angles(noise, labels))@self.z_signs
        return self.decoder(features).reshape(len(noise), self.channels, self.image_size, self.image_size)

    def circuit_report(self, noise, labels):
        self._inputs(noise, labels)
        return self.quantum.explain(self.encoded_angles(noise[:1], labels[:1])[0])


class ClassicalImageGenerator(_Generator):
    """Conditional MLP or convolutional baseline, with the same latent prior."""
    def __init__(self, image_size=8, channels=1, latent_dim=4, num_classes=2, hidden=64, seed=7,
                 architecture='mlp'):
        super().__init__(); self._setup(image_size, channels, latent_dim, num_classes)
        seed = _seed(seed)
        image_size, channels, latent_dim, num_classes = self.image_size, self.channels, self.latent_dim, self.num_classes
        hidden = _bounded_int(hidden, 'hidden', 512)
        architecture = _architecture(architecture, image_size)
        self.config = dict(kind='classical', image_size=image_size, channels=channels, latent_dim=latent_dim,
                           num_classes=num_classes, hidden=hidden, seed=seed, architecture=architecture)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.network = (_conv_decoder(latent_dim+num_classes, image_size, channels, hidden)
                if architecture == 'conv' else nn.Sequential(nn.Linear(latent_dim+num_classes, hidden), nn.LeakyReLU(.2),
                    nn.Linear(hidden, hidden), nn.LeakyReLU(.2), nn.Linear(hidden, channels*image_size**2), nn.Sigmoid()))

    def forward(self, noise, labels):
        self._inputs(noise, labels)
        code = torch.nn.functional.one_hot(labels, self.num_classes).to(noise.dtype)
        return self.network(torch.cat([noise, code], dim=1)).reshape(len(noise), self.channels, self.image_size, self.image_size)


def build_generator(config):
    config = dict(config); kind = config.pop('kind', None)
    classes = dict(patch=QuantumPatchGenerator, hybrid=HybridImageGenerator, classical=ClassicalImageGenerator)
    if kind not in classes: raise ValueError('Unknown generator kind')
    return classes[kind](**config)
