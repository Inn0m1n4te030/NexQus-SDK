"""Optional CPU tests: python -m unittest discover -s tests/image_gan -v."""
import tempfile
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch
import numpy as np
import torch
from PIL import Image
from nexqus.image_gan import (QuantumProbabilityLayer, QuantumPatchGenerator,
    HybridImageGenerator, ClassicalImageGenerator, ImageGAN, ImageDataset,
    synthetic_bars, save_image_grid, load_cifar10_cats_dogs)
from nexqus.image_models import _Probabilities


class ImageGANTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.noise = torch.tensor([[.2, -.4], [.8, .1]])
        self.labels = torch.tensor([0, 1])

    def test_quantum_bridge_gradcheck_weights_and_inputs(self):
        layer = QuantumProbabilityLayer(n_qubits=2, depth=2).double()
        weights = layer.weights.detach().clone().requires_grad_()
        angles = self.noise.double().requires_grad_()
        self.assertTrue(torch.autograd.gradcheck(
            lambda w, z: _Probabilities.apply(w, z, layer), (weights, angles),
            eps=1e-6, atol=2e-6, rtol=1e-4))
        expected = layer.function(layer.bindings(weights.detach(), angles.detach()[0]))
        np.testing.assert_allclose(layer(angles)[0].detach(), expected, atol=1e-12)
        torch.testing.assert_close(layer(angles).sum(1), torch.ones(2, dtype=torch.double))

    def test_shapes_conditionality_and_gradient_paths(self):
        for kind in (QuantumPatchGenerator, HybridImageGenerator, ClassicalImageGenerator):
            with self.subTest(kind=kind.__name__):
                model = kind(image_size=4, channels=3, latent_dim=2)
                output = model(self.noise, self.labels)
                self.assertEqual(output.shape, (2, 3, 4, 4))
                self.assertTrue(torch.isfinite(output).all())
                self.assertTrue(((output >= 0) & (output <= 1)).all())
                self.assertFalse(torch.allclose(output, model(self.noise, self.labels.flip(0))))
                self.assertFalse(torch.allclose(output, model(-self.noise, self.labels)))
                coefficients = torch.linspace(.1, 1, output.numel()).reshape_as(output)
                (output*coefficients).sum().backward()
                for name, parameter in model.named_parameters():
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                if kind != ClassicalImageGenerator:
                    self.assertGreater(float(model.encoder.weight.grad.abs().sum()), 1e-6)
                    self.assertGreater(sum(float(q.weights.grad.abs().sum()) for q in model.quantum_layers()), 1e-6)
                if kind == HybridImageGenerator:
                    self.assertGreater(float(model.decoder[0].weight.grad.abs().sum()), 1e-6)

    def test_patch_layout_and_normalization(self):
        model = QuantumPatchGenerator(image_size=4, channels=3, latent_dim=2, patch_size=2)
        angles = model.encoded_angles(self.noise, self.labels)
        image = model(self.noise, self.labels)
        for index, circuit in enumerate(model.circuits):
            channel, spatial = divmod(index, 4); row, col = divmod(spatial, 2)
            selected = circuit(angles).reshape(2, 4, 2)[:, :, 0]
            expected = (selected/selected.amax(1, keepdim=True)).reshape(2, 2, 2)
            torch.testing.assert_close(image[:, channel, row*2:row*2+2, col*2:col*2+2], expected)

    def test_training_and_exact_checkpoint_continuation(self):
        data = synthetic_bars(samples=8, image_size=4)
        for kind in (QuantumPatchGenerator, HybridImageGenerator, ClassicalImageGenerator):
            with self.subTest(kind=kind.__name__), tempfile.TemporaryDirectory() as directory:
                trainer = ImageGAN(kind(image_size=4, latent_dim=2), seed=17)
                original = {name: p.detach().clone() for name, p in trainer.generator.named_parameters()}
                trainer.fit(data, batch_size=2, max_steps=1)
                self.assertEqual(trainer.steps, 1)
                self.assertTrue(any(not torch.equal(original[n], p) for n, p in trainer.generator.named_parameters()))
                path = Path(directory)/'checkpoint.pt'; trainer.save(path)
                restored = ImageGAN.load(path)
                torch.testing.assert_close(trainer.sample(4), restored.sample(4), rtol=0, atol=0)
                noise_state = trainer.noise_rng.get_state().clone(); trainer.sample(3)
                torch.testing.assert_close(noise_state, trainer.noise_rng.get_state(), rtol=0, atol=0)
                trainer.fit(data, batch_size=2, max_steps=1)
                restored.fit(data, batch_size=2, max_steps=1)
                self.assertEqual(trainer.history, restored.history)
                for n, p in trainer.generator.state_dict().items():
                    torch.testing.assert_close(p, restored.generator.state_dict()[n], rtol=0, atol=0)
                for n, p in trainer.discriminator.state_dict().items():
                    torch.testing.assert_close(p, restored.discriminator.state_dict()[n], rtol=0, atol=0)
                with self.assertRaisesRegex(ValueError, 'batch size'): restored.fit(data, batch_size=1)
                changed = synthetic_bars(samples=8, image_size=4, seed=9)
                with self.assertRaisesRegex(ValueError, 'Dataset changed'): restored.fit(changed, batch_size=2)

    def test_counterfactual_restores_model_and_explains_circuit(self):
        for kind in (QuantumPatchGenerator, HybridImageGenerator):
            model = kind(image_size=4, latent_dim=2)
            state = {n: p.clone() for n, p in model.state_dict().items()}
            effect = model.parameter_effect(self.noise[:1], self.labels[:1])
            self.assertTrue(model.training)
            self.assertGreater(float(effect['sensitivity'].abs().sum()), 0)
            torch.testing.assert_close(effect['sensitivity'], (effect['plus']-effect['minus'])/.4)
            for name, p in model.state_dict().items(): torch.testing.assert_close(p, state[name], rtol=0, atol=0)
            report = model.circuit_report(self.noise[:1], self.labels[:1]).to_html()
            self.assertIn('latent angles', report)
            with self.assertRaises(ValueError): model.parameter_effect(self.noise, self.labels)
        patches = QuantumPatchGenerator(image_size=8, latent_dim=2)
        effect = patches.parameter_effect(self.noise[:1], self.labels[:1], index=15)
        difference = effect['plus']-effect['minus']
        self.assertGreater(float(difference[:, :, :4, 4:].abs().sum()), 0)
        self.assertEqual(float(difference[:, :, 4:, :].abs().sum()), 0)
        self.assertEqual(float(difference[:, :, :4, :4].abs().sum()), 0)

    def test_folder_split_and_image_export(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for label in ('cats', 'dogs'):
                (root/label).mkdir()
                for index in range(5):
                    pixels = np.full((12, 8, 3), index*20+(80 if label == 'dogs' else 0), np.uint8)
                    Image.fromarray(pixels).save(root/label/f'{index}.png')
            data = ImageDataset.from_folder(root, image_size=4, max_images=8)
            self.assertEqual(data.images.shape, (8, 3, 4, 4)); self.assertEqual(data.class_names, ('cats', 'dogs'))
            selected = ImageDataset.from_folder(root, image_size=4, max_images=4, selection_seed=19)
            self.assertEqual(selected.fingerprint,
                ImageDataset.from_folder(root, image_size=4, max_images=4, selection_seed=19).fingerprint)
            self.assertNotEqual(selected.fingerprint,
                ImageDataset.from_folder(root, image_size=4, max_images=4).fingerprint)
            self.assertEqual(selected.labels.tolist(), [0, 0, 1, 1])
            train, validation = data.split(seed=2)
            self.assertEqual(len(train)+len(validation), 8)
            self.assertFalse(set(train.images[:, 0, 0, 0].tolist()) & set(validation.images[:, 0, 0, 0].tolist()))
            self.assertEqual(train.fingerprint, data.split(seed=2)[0].fingerprint)
            save_image_grid(data.images, root/'grid.png', columns=4, scale=2)
            with Image.open(root/'grid.png') as image: self.assertEqual(image.size, (42, 22))
            (root/'cats'/'0.png').write_bytes(b'broken')
            with self.assertRaisesRegex(ValueError, 'Cannot read image'): ImageDataset.from_folder(root, image_size=4)

    def test_cifar_mapping_without_download(self):
        class FakeCIFAR:
            def __init__(self, **kwargs):
                self.targets = [0, 3, 5, 3, 5]; self.options = kwargs
                ImageGANTests.cifar_options = kwargs
            def __getitem__(self, index):
                return Image.new('RGB', (32, 32), (index*20, 0, 0)), self.targets[index]
        datasets = types.ModuleType('torchvision.datasets'); datasets.CIFAR10 = FakeCIFAR
        vision = types.ModuleType('torchvision'); vision.datasets = datasets
        with patch.dict(sys.modules, {'torchvision': vision, 'torchvision.datasets': datasets}):
            data = load_cifar10_cats_dogs('unused', image_size=4, max_images=4, train=False)
        self.assertEqual(data.class_names, ('cat', 'dog'))
        self.assertEqual(data.labels.tolist(), [0, 0, 1, 1])
        self.assertFalse(self.cifar_options['download']); self.assertFalse(self.cifar_options['train'])

    def test_validation_and_rng_isolation(self):
        state = torch.random.get_rng_state()
        models = [QuantumPatchGenerator(), HybridImageGenerator(), ClassicalImageGenerator()]
        ImageGAN(models[0]); synthetic_bars()
        torch.testing.assert_close(state, torch.random.get_rng_state(), rtol=0, atol=0)
        with self.assertRaises(ValueError): QuantumPatchGenerator(image_size=64, channels=3, patch_size=2)
        with self.assertRaises(ValueError): QuantumProbabilityLayer(n_qubits=9)
        with self.assertRaises(ValueError): models[0](torch.zeros(1, 4), torch.tensor([2]))
        with self.assertRaises(ValueError): models[0](torch.zeros(1, 4, dtype=torch.double), torch.tensor([0]))
        with self.assertRaises(ValueError): QuantumProbabilityLayer()(torch.full((1, 4), float('nan')))
        with self.assertRaises(ValueError): synthetic_bars(samples=10001)
        with self.assertRaises(ValueError): models[2].parameter_effect(torch.zeros(1, 4), torch.tensor([0]))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'invalid.pt'; torch.save({'schema_version': 99}, path)
            with self.assertRaises(ValueError): ImageGAN.load(path)
            model = HybridImageGenerator(image_size=np.int64(4), n_qubits=np.int64(2), seed=np.int64(3))
            trainer = ImageGAN(model, seed=np.int64(3)); trainer.save(path)
            self.assertEqual(ImageGAN.load(path).generator.config, model.config)


if __name__ == '__main__': unittest.main()
