"""Conditional image GAN training, deterministic continuation and safe loading.

Optional import: from nexqus.image_gan import ... . Core nexqus does not import
PyTorch, Pillow or this module. Quantum circuit execution is local CPU only.
"""
import os
from pathlib import Path
import tempfile
try:
    import torch
    from torch import nn
except ImportError as exc:
    raise ImportError('Install the source checkout with .[image-gan] to use image GANs') from exc
from .circuit import positive_integer
from .parameters import real_scalar
from .image_models import (QuantumProbabilityLayer, QuantumPatchGenerator,
    HybridImageGenerator, ClassicalImageGenerator, build_generator, _bounded_int, _seed)
from .image_data import ImageDataset, synthetic_bars, load_cifar10_cats_dogs, save_image_grid


class ImageDiscriminator(nn.Module):
    def __init__(self, image_size, channels, num_classes, hidden=64, seed=7):
        super().__init__(); hidden = _bounded_int(hidden, 'hidden', 512)
        self.num_classes = num_classes
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.network = nn.Sequential(nn.Linear(channels*image_size**2+num_classes, hidden),
                nn.LeakyReLU(.2), nn.Linear(hidden, hidden), nn.LeakyReLU(.2), nn.Linear(hidden, 1))

    def forward(self, images, labels):
        classes = torch.nn.functional.one_hot(labels, self.num_classes).to(images.dtype)
        return self.network(torch.cat([images.flatten(1), classes], dim=1)).flatten()


class ImageGAN:
    """Alternating balanced BCE discriminator / non-saturating generator loss.

    All computation is CPU float32. Training uses local random generators;
    sampling does not change the training RNG. fit can stop/resume mid-epoch
    with the same dataset fingerprint and batch size.
    """
    def __init__(self, generator, *, learning_rate=.002, discriminator_hidden=64, seed=7):
        seed = _seed(seed); discriminator_hidden = _bounded_int(discriminator_hidden, 'discriminator_hidden', 512)
        if not isinstance(generator, (QuantumPatchGenerator, HybridImageGenerator, ClassicalImageGenerator)):
            raise TypeError('Expected a supported image generator')
        if any(p.device.type != 'cpu' or p.dtype != torch.float32 for p in generator.parameters()):
            raise ValueError('ImageGAN training requires CPU float32 model parameters')
        learning_rate = real_scalar(learning_rate)
        if learning_rate <= 0: raise ValueError('learning_rate must be positive')
        self.generator = generator
        self.discriminator = ImageDiscriminator(generator.image_size, generator.channels,
            generator.num_classes, discriminator_hidden, seed)
        self.generator_optimizer = torch.optim.Adam(generator.parameters(), lr=learning_rate, betas=(.5,.999))
        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=learning_rate, betas=(.5,.999))
        self.config = dict(learning_rate=learning_rate, discriminator_hidden=discriminator_hidden, seed=seed)
        self.noise_rng = torch.Generator().manual_seed(seed+101)
        self.order_rng = torch.Generator().manual_seed(seed+202)
        self.epoch = 0; self.steps = 0; self.history = []
        self._order = None; self._cursor = 0; self._batch_size = None; self._fingerprint = None

    def _step(self, real, labels):
        g, d = self.generator, self.discriminator
        g.train(); d.train()
        noise = torch.randn(len(real), g.latent_dim, generator=self.noise_rng)
        with torch.no_grad(): fake = g(noise, labels)
        self.discriminator_optimizer.zero_grad(set_to_none=True)
        real_logits, fake_logits = d(real, labels), d(fake, labels)
        loss_d = .5*(torch.nn.functional.softplus(-real_logits).mean()+torch.nn.functional.softplus(fake_logits).mean())
        if not torch.isfinite(loss_d): raise FloatingPointError('Discriminator loss is not finite')
        loss_d.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in d.parameters()):
            raise FloatingPointError('Discriminator gradient is not finite')
        self.discriminator_optimizer.step()
        self.generator_optimizer.zero_grad(set_to_none=True)
        for p in d.parameters(): p.requires_grad_(False)
        try:
            noise = torch.randn(len(real), g.latent_dim, generator=self.noise_rng)
            fake = g(noise, labels)
            loss_g = torch.nn.functional.softplus(-d(fake, labels)).mean()
            if not torch.isfinite(loss_g): raise FloatingPointError('Generator loss is not finite')
            loss_g.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in g.parameters()):
                raise FloatingPointError('Generator gradient is not finite')
            self.generator_optimizer.step()
        finally:
            for p in d.parameters(): p.requires_grad_(True)
        return dict(discriminator_loss=float(loss_d.detach()), generator_loss=float(loss_g.detach()),
                    generated_mean=float(fake.detach().mean()))

    def fit(self, dataset, *, epochs=1, batch_size=8, max_steps=None, callback=None):
        if not isinstance(dataset, ImageDataset): raise TypeError('Expected ImageDataset')
        epochs = positive_integer(epochs, 'epochs'); batch_size = _bounded_int(batch_size, 'batch_size', 128)
        if max_steps is not None: max_steps = positive_integer(max_steps, 'max_steps')
        g = self.generator
        if dataset.images.shape[1:] != (g.channels, g.image_size, g.image_size) or len(dataset.class_names) != g.num_classes:
            raise ValueError('Dataset shape/classes do not match the generator')
        fingerprint = dataset.fingerprint
        if self._fingerprint is not None and self._fingerprint != fingerprint:
            raise ValueError('Dataset changed; create a new trainer instead of silently resuming on different data')
        if self._order is not None and self._batch_size != batch_size:
            raise ValueError('Keep the same batch size when continuing a partial epoch')
        self._fingerprint = fingerprint; self._batch_size = batch_size
        if self._order is not None and (self._order.shape != (len(dataset),) or self._cursor < 0
                or self._cursor >= len(dataset) or not torch.equal(torch.sort(self._order).values, torch.arange(len(dataset)))):
            raise ValueError('Invalid saved data-order state')
        target_epoch = self.epoch+epochs; start_steps = self.steps
        while self.epoch < target_epoch:
            if self._order is None: self._order = torch.randperm(len(dataset), generator=self.order_rng); self._cursor = 0
            indices = self._order[self._cursor:self._cursor+batch_size]
            record = self._step(dataset.images[indices], dataset.labels[indices])
            record.update(step=self.steps+1, epoch=self.epoch)
            self.steps += 1; self._cursor += len(indices); self.history.append(record)
            if self._cursor == len(dataset): self.epoch += 1; self._order = None; self._cursor = 0
            if callback is not None: callback(dict(record))
            if max_steps is not None and self.steps-start_steps >= max_steps: break
        return self

    def sample(self, count=16, *, labels=None, seed=0):
        count = _bounded_int(count, 'count', 128)
        if labels is None: labels = torch.arange(count, dtype=torch.long) % self.generator.num_classes
        if not isinstance(labels, torch.Tensor) or labels.shape != (count,): raise ValueError('Supply one label per image')
        noise = torch.randn(count, self.generator.latent_dim, generator=torch.Generator().manual_seed(seed))
        previous = self.generator.training; self.generator.eval()
        try:
            with torch.no_grad(): return self.generator(noise, labels).clone()
        finally: self.generator.train(previous)

    def save(self, filename):
        """Atomic tensor/primitives checkpoint; includes optimizers and RNG/data order."""
        filename = Path(filename); filename.parent.mkdir(parents=True, exist_ok=True)
        state = dict(schema_version=1, generator_config=self.generator.config, trainer_config=self.config,
            generator_state=self.generator.state_dict(), discriminator_state=self.discriminator.state_dict(),
            generator_optimizer=self.generator_optimizer.state_dict(), discriminator_optimizer=self.discriminator_optimizer.state_dict(),
            noise_rng=self.noise_rng.get_state(), order_rng=self.order_rng.get_state(),
            epoch=self.epoch, steps=self.steps, history=self.history, order=self._order, cursor=self._cursor,
            batch_size=self._batch_size, fingerprint=self._fingerprint)
        with tempfile.NamedTemporaryFile(dir=filename.parent, prefix=filename.name+'.', suffix='.tmp', delete=False) as f:
            temporary = Path(f.name)
        try:
            torch.save(state, temporary); os.replace(temporary, filename)
        finally:
            if temporary.exists(): temporary.unlink()

    @classmethod
    def load(cls, filename):
        """Load only tensor/primitives data; no custom model pickle objects."""
        state = torch.load(filename, map_location='cpu', weights_only=True)
        if not isinstance(state, dict) or state.get('schema_version') != 1: raise ValueError('Unsupported checkpoint')
        trainer = cls(build_generator(state['generator_config']), **state['trainer_config'])
        trainer.generator.load_state_dict(state['generator_state']); trainer.discriminator.load_state_dict(state['discriminator_state'])
        trainer.generator_optimizer.load_state_dict(state['generator_optimizer'])
        trainer.discriminator_optimizer.load_state_dict(state['discriminator_optimizer'])
        trainer.noise_rng.set_state(state['noise_rng']); trainer.order_rng.set_state(state['order_rng'])
        trainer.epoch = int(state['epoch']); trainer.steps = int(state['steps']); trainer.history = state['history']
        trainer._order = state['order']; trainer._cursor = int(state['cursor'])
        trainer._batch_size = state['batch_size']; trainer._fingerprint = state['fingerprint']
        return trainer


__all__ = ['QuantumProbabilityLayer', 'QuantumPatchGenerator', 'HybridImageGenerator',
           'ClassicalImageGenerator', 'ImageDiscriminator', 'ImageGAN', 'ImageDataset',
           'synthetic_bars', 'load_cifar10_cats_dogs', 'save_image_grid']
