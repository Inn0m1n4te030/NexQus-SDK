"""Regularized gradient paths, R1 scaling, diagnostics and exact resume."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from nexqus.image_gan import ImageGAN, HybridImageGenerator, ClassicalImageGenerator, synthetic_bars
from nexqus.image_metrics import image_diagnostics, nearest_training_images, hybrid_diagnostics


class StabilityTests(unittest.TestCase):
    device = 'cpu'

    def setUp(self):
        torch.set_num_threads(1)
        self.context = torch.backends.cudnn.flags(enabled=True, benchmark=False, deterministic=True)
        self.context.__enter__(); self.addCleanup(self.context.__exit__, None, None, None)

    def trainer(self, hybrid=False, architecture='conv', size=8, channels=1):
        options = dict(image_size=size, channels=channels, latent_dim=4, hidden=16, architecture=architecture)
        model = (HybridImageGenerator(**options, readout='probabilities', angle_map='linear', latent_skip=True)
                 if hybrid else ClassicalImageGenerator(**options))
        return ImageGAN(model, device=self.device, discriminator_hidden=16, training='regularized',
            generator_learning_rate=.0002, discriminator_learning_rate=.0001,
            r1_gamma=.5, r1_every=2, diff_augment=True, ema_decay=.9, horizontal_flip=True)

    def assert_state(self, left, right):
        self.assertEqual(set(left), set(right))
        for key in left: torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)

    def test_regularized_gradients_ema_and_exact_resume(self):
        data = synthetic_bars(samples=12, image_size=8)
        for hybrid in (False, True):
            with self.subTest(hybrid=hybrid), tempfile.TemporaryDirectory() as directory:
                trainer = self.trainer(hybrid)
                old = {n:p.detach().clone() for n,p in trainer.generator.named_parameters()}
                trainer.fit(data, batch_size=4, max_steps=1)
                self.assertGreater(trainer.history[-1]['r1_penalty'], 0)
                for name, p in trainer.ema_generator.named_parameters():
                    torch.testing.assert_close(p, .9*old[name]+.1*dict(trainer.generator.named_parameters())[name], rtol=1e-5, atol=1e-7)
                for name, value in trainer.generator.named_buffers():
                    if name.endswith('num_batches_tracked'): self.assertEqual(int(value), 1)
                if hybrid: self.assertGreater(float(trainer.generator.quantum.weights.grad.abs().sum()), 0)
                noise = trainer.noise_rng.get_state().clone(); augmentation = trainer.augmentation_rng.get_state().clone()
                state = {n:p.clone() for n,p in trainer.generator.state_dict().items()}; trainer.sample(4)
                self.assert_state(state, trainer.generator.state_dict())
                self.assertTrue(torch.equal(noise, trainer.noise_rng.get_state()))
                self.assertTrue(torch.equal(augmentation, trainer.augmentation_rng.get_state()))
                path = Path(directory)/'checkpoint.pt'; trainer.save(path); restored = ImageGAN.load(path)
                trainer.fit(data, batch_size=4, max_steps=2); restored.fit(data, batch_size=4, max_steps=2)
                self.assertEqual(trainer.history, restored.history); self.assertEqual(trainer.history[1]['r1_penalty'], 0)
                for key in ('generator','discriminator','ema_generator'):
                    self.assert_state(getattr(trainer,key).state_dict(), getattr(restored,key).state_dict())
                torch.testing.assert_close(trainer.sample(4), restored.sample(4), rtol=0, atol=0)
                self.assertFalse(torch.equal(trainer.sample(4), trainer.sample(4, use_ema=False)))

    def test_legacy_schema3_compatibility_and_rng_isolation(self):
        rng = torch.random.get_rng_state().clone(); data = synthetic_bars(samples=8, image_size=8)
        for constructor in (ClassicalImageGenerator, HybridImageGenerator):
            trainer = ImageGAN(constructor(image_size=8, latent_dim=4, hidden=16, architecture='conv'),
                               discriminator_hidden=16, device=self.device)
            self.assertTrue(torch.equal(rng, torch.random.get_rng_state())); trainer.fit(data, batch_size=2, max_steps=1)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory)/'old.pt'; trainer.save(path)
                state = torch.load(path, map_location='cpu', weights_only=True)
                state['schema_version'] = 3; state.pop('ema_state')
                for key in ('training','generator_learning_rate','discriminator_learning_rate','r1_gamma','r1_every','diff_augment','ema_decay'):
                    state['trainer_config'].pop(key)
                for key in ('readout','angle_map','latent_skip'): state['generator_config'].pop(key, None)
                torch.save(state,path); loaded = ImageGAN.load(path)
                self.assertEqual(loaded.config['training'], 'legacy')
                trainer.fit(data,batch_size=2,max_steps=1); loaded.fit(data,batch_size=2,max_steps=1)
                self.assertEqual(trainer.history,loaded.history); self.assert_state(trainer.generator.state_dict(),loaded.generator.state_dict())

    def test_analytic_r1_scale(self):
        class SumDiscriminator(torch.nn.Module):
            def __init__(self, device):
                super().__init__(); self.weight = torch.nn.Parameter(torch.tensor(.01, device=device))
            def forward(self, images, labels): return self.weight*images.flatten(1).sum(1)
        trainer = self.trainer(); trainer.retune(r1_gamma=2, r1_every=4)
        trainer.discriminator = SumDiscriminator(self.device)
        trainer.discriminator_optimizer = torch.optim.SGD(trainer.discriminator.parameters(), lr=.001)
        record = trainer._step(torch.full((2,1,8,8), .3, device=self.device), torch.tensor([0,1], device=self.device))
        self.assertAlmostEqual(record['r1_penalty'], .5*2*4*64*.01**2, places=6)

    def test_symmetric_augmentation_and_retune_preserves_state(self):
        trainer = self.trainer()
        with patch.object(trainer, '_augment', wraps=trainer._augment) as augment:
            trainer.fit(synthetic_bars(samples=12, image_size=8), batch_size=4, max_steps=1)
            self.assertEqual(augment.call_count, 3)
        state = {n:p.clone() for n,p in trainer.generator.state_dict().items()}
        moments = {id(p):v['exp_avg'].clone() for p,v in trainer.generator_optimizer.state.items()}
        cursor = trainer._cursor; order = trainer._order.clone()
        trainer.retune(generator_learning_rate=.0001, discriminator_learning_rate=.00005,r1_gamma=1, diff_augment=True, ema_decay=.9)
        self.assert_state(state, trainer.generator.state_dict()); self.assertEqual(cursor, trainer._cursor)
        self.assertTrue(torch.equal(order, trainer._order))
        for p,v in trainer.generator_optimizer.state.items(): self.assertTrue(torch.equal(moments[id(p)],v['exp_avg']))
        image = torch.ones(2,1,8,8, device=self.device, requires_grad=True)
        trainer._augment(image).sum().backward(); self.assertGreater(float(image.grad.abs().sum()), 0)
        with self.assertRaises(ValueError): trainer.retune(training='legacy', r1_gamma=1)
        with self.assertRaises(ValueError): trainer.retune(generator_learning_rate=0)

    def test_diagnostics_detect_identical_outputs_and_preserve_model(self):
        data = synthetic_bars(samples=12, image_size=8)
        images = data.images[:1].repeat(4,1,1,1); labels = data.labels[:1].repeat(4)
        stats = image_diagnostics(images, labels, data)[data.class_names[int(labels[0])]]
        self.assertEqual(stats['near_duplicate_pair_fraction'], 1); self.assertEqual(stats['mean_pairwise_pixel_mae'], 0)
        nearest, info = nearest_training_images(images, labels, data)
        self.assertEqual(info['distances'], [0]*4); torch.testing.assert_close(images, nearest)
        self.assertIsNone(image_diagnostics(images[:1],labels[:1],data)[data.class_names[int(labels[0])]]['closest_pair_pixel_mae'])
        model = self.trainer(True).generator; state = {n:p.clone() for n,p in model.state_dict().items()}
        noise = torch.randn(8,4, generator=torch.Generator().manual_seed(5)).to(self.device)
        result = hybrid_diagnostics(model, noise, torch.zeros(8, dtype=torch.long, device=self.device))
        self.assertEqual(result['quantum_feature_dimension'],16); self.assertTrue(result['latent_skip'])
        self.assertIsNone(result['classes']['0']['tanh_saturation_fraction']); self.assertGreater(result['classes']['0']['mean_quantum_feature_std'],0)
        self.assert_state(state, model.state_dict())

    def test_dcgan_gradients_and_checkpoint(self):
        data = synthetic_bars(samples=8, image_size=32, channels=3)
        for hybrid in (False, True):
            with self.subTest(hybrid=hybrid), tempfile.TemporaryDirectory() as directory:
                trainer = self.trainer(hybrid, architecture='dcgan', size=32, channels=3)
                noise = torch.randn(2,4,device=self.device,requires_grad=True); labels = torch.tensor([0,1],device=self.device)
                output = trainer.generator(noise,labels)
                self.assertEqual(output.shape,(2,3,32,32)); self.assertTrue(((output>=0)&(output<=1)).all())
                (output*torch.linspace(.1,1,output.numel(),device=self.device).reshape_as(output)).mean().backward()
                self.assertGreater(float(noise.grad.abs().sum()),0)
                for p in trainer.generator.parameters(): self.assertTrue(torch.isfinite(p.grad).all())
                trainer.fit(data,batch_size=4,max_steps=1); path=Path(directory)/'checkpoint.pt'; trainer.save(path); restored=ImageGAN.load(path)
                self.assertEqual(restored.discriminator.architecture,'dcgan')
                torch.testing.assert_close(trainer.sample(4),restored.sample(4),rtol=0,atol=0)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
class CUDAStabilityTests(StabilityTests):
    device = 'cuda:0'


if __name__ == '__main__': unittest.main()
