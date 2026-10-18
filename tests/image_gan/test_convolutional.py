"""Spatial models: real gradient paths, state restoration and portable checkpoints."""
import os
from pathlib import Path
import tempfile
import unittest
import torch
from nexqus.image_gan import ClassicalImageGenerator, HybridImageGenerator, ImageGAN, synthetic_bars


class ConvolutionalTests(unittest.TestCase):
    device = 'cpu'

    def setUp(self):
        torch.set_num_threads(1)
        flags = torch.backends.cudnn.flags(deterministic=True, benchmark=False)
        flags.__enter__()
        self.addCleanup(flags.__exit__, None, None, None)

    def test_spatial_shapes_and_gradients(self):
        for constructor in (ClassicalImageGenerator, HybridImageGenerator):
            for size in (4, 16, 64):
                with self.subTest(model=constructor.__name__, size=size):
                    model = constructor(image_size=size, channels=3, latent_dim=4,
                                        hidden=32, architecture='conv').to(self.device)
                    noise = torch.randn(2, 4, device=self.device, requires_grad=True)
                    labels = torch.tensor([0, 1], device=self.device)
                    output = model(noise, labels)
                    self.assertEqual(output.shape, (2, 3, size, size))
                    self.assertTrue(((output >= 0) & (output <= 1)).all())
                    (output*torch.linspace(0, 1, output.numel(), device=self.device).reshape_as(output)).sum().backward()
                    self.assertGreater(float(noise.grad.abs().sum()), 0)
                    for name, parameter in model.named_parameters():
                        self.assertIsNotNone(parameter.grad, name)
                        self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                    if constructor == HybridImageGenerator:
                        self.assertGreater(float(model.quantum.weights.grad.abs().sum()), 1e-6)
                    model.eval()
                    self.assertFalse(torch.allclose(model(noise, labels), model(noise, labels.flip(0))))
                    with self.assertRaisesRegex(ValueError, 'powers of two'):
                        constructor(image_size=12, architecture='conv')

    def test_augmented_training_and_exact_resume(self):
        data = synthetic_bars(samples=12, image_size=16)
        for constructor in (ClassicalImageGenerator, HybridImageGenerator):
            with self.subTest(model=constructor.__name__), tempfile.TemporaryDirectory() as folder:
                trainer = ImageGAN(constructor(image_size=16, hidden=32, architecture='conv'),
                    discriminator_hidden=32, horizontal_flip=True, device=self.device)
                self.assertEqual(trainer.discriminator.architecture, 'conv')
                original = {name: p.detach().clone() for name, p in trainer.generator.named_parameters()}
                trainer.fit(data, batch_size=4, max_steps=2)
                self.assertTrue(all(not torch.equal(original[n], p) for n, p in trainer.generator.named_parameters()))
                # Sampling and explanations must not change BatchNorm buffers or the augmentation RNG.
                state = {n: p.clone() for n, p in trainer.generator.state_dict().items()}
                aug = trainer.augmentation_rng.get_state().clone()
                trainer.sample(3)
                if constructor == HybridImageGenerator:
                    effect = trainer.generator.parameter_effect(torch.zeros(1, 4, device=self.device),
                                                                torch.zeros(1, dtype=torch.long, device=self.device))
                    self.assertGreater(float(effect['sensitivity'].abs().sum()), 0)
                for name, p in trainer.generator.state_dict().items():
                    torch.testing.assert_close(p, state[name], atol=0, rtol=0)
                torch.testing.assert_close(aug, trainer.augmentation_rng.get_state(), atol=0, rtol=0)
                path = Path(folder)/'model.pt'; trainer.save(path)
                restored = ImageGAN.load(path, device=self.device)
                for model in (trainer, restored): model.fit(data, batch_size=4, max_steps=2)
                self.assertEqual(trainer.history, restored.history)
                for a, b in ((trainer.generator, restored.generator), (trainer.discriminator, restored.discriminator)):
                    for name, p in a.state_dict().items():
                        torch.testing.assert_close(p, b.state_dict()[name], atol=0, rtol=0)
                torch.testing.assert_close(trainer.sample(4), restored.sample(4), atol=0, rtol=0)

    def test_legacy_schema_and_rng_isolation(self):
        before = torch.random.get_rng_state().clone()
        ImageGAN(ClassicalImageGenerator(image_size=16, architecture='conv'))
        torch.testing.assert_close(before, torch.random.get_rng_state(), atol=0, rtol=0)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'old.pt'
            trainer = ImageGAN(ClassicalImageGenerator(image_size=4), device=self.device)
            trainer.fit(synthetic_bars(samples=8, image_size=4), batch_size=2, max_steps=1)
            trainer.save(path)
            state = torch.load(path, weights_only=True)
            state['schema_version'] = 2
            state.pop('augmentation_rng')
            state['generator_config'].pop('architecture')
            for key in ('discriminator_architecture', 'horizontal_flip'): state['trainer_config'].pop(key)
            torch.save(state, path)
            restored = ImageGAN.load(path, device=self.device)
            self.assertEqual(restored.generator.config['architecture'], 'mlp')
            self.assertEqual(restored.discriminator.architecture, 'mlp')
            torch.testing.assert_close(trainer.sample(4), restored.sample(4), atol=0, rtol=0)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA GPU unavailable')
class CUDAConvolutionalTests(ConvolutionalTests):
    device = os.environ.get('NEXQUS_TEST_CUDA_DEVICE', 'cuda:0')


if __name__ == '__main__': unittest.main()
