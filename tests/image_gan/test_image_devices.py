import os
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import torch
from nexqus.image_gan import QuantumPatchGenerator,HybridImageGenerator,ClassicalImageGenerator,ImageGAN,synthetic_bars


class ImageDeviceTests(unittest.TestCase):
    device='cpu'

    def test_batched_native_forward_backward_and_training(self):
        data=synthetic_bars(samples=8,image_size=4)
        noise=torch.tensor([[.2,-.3],[.7,.8]],device=self.device)
        labels=torch.tensor([0,1],device=self.device)
        for constructor in (QuantumPatchGenerator,HybridImageGenerator,ClassicalImageGenerator):
            with self.subTest(model=constructor.__name__):
                cpu=constructor(image_size=4,channels=3,latent_dim=2)
                target=constructor(image_size=4,channels=3,latent_dim=2).to(self.device)
                actual=target(noise,labels)
                torch.testing.assert_close(actual.cpu(),cpu(noise.cpu(),labels.cpu()),atol=3e-6,rtol=3e-5)
                actual.square().sum().backward()
                for p in target.parameters():
                    self.assertIsNotNone(p.grad); self.assertEqual(p.grad.device.type,torch.device(self.device).type)
                if constructor is not ClassicalImageGenerator:
                    layer=target.quantum_layers()[0]
                    angles=torch.full((2,layer.n_qubits),.3,device=self.device,requires_grad=True)
                    with patch.object(layer.function,'_evaluate',side_effect=AssertionError('NumPy execution in training')):
                        layer(angles).square().sum().backward()
                    effect=target.parameter_effect(noise[:1],labels[:1])
                    self.assertEqual(effect['plus'].device.type,torch.device(self.device).type)
                    self.assertIn('Image generator',target.circuit_report(noise[:1],labels[:1]).to_html())
                trainer=ImageGAN(constructor(image_size=4,latent_dim=2),device=self.device)
                trainer.fit(data,batch_size=2,max_steps=1)
                with tempfile.TemporaryDirectory() as folder:
                    file=Path(folder)/'image.pt'; trainer.save(file)
                    restored=ImageGAN.load(file,device=self.device)
                    trainer.fit(data,batch_size=2,max_steps=1); restored.fit(data,batch_size=2,max_steps=1)
                    self.assertEqual(trainer.history,restored.history)
                    torch.testing.assert_close(trainer.sample(4),restored.sample(4),atol=0,rtol=0)
                    for state in restored.generator_optimizer.state.values():
                        if 'exp_avg' in state: self.assertEqual(state['exp_avg'].device,restored.device)
                    if self.device!='cpu':
                        with self.assertRaisesRegex(ValueError,'RNG'): ImageGAN.load(file,device='cpu')
                        portable=ImageGAN.load(file,device='cpu',reset_rng=True)
                        self.assertEqual(portable.steps,trainer.steps-1)


@unittest.skipUnless(torch.cuda.is_available(),'CUDA GPU unavailable')
class CUDAImageDeviceTests(ImageDeviceTests):
    device=os.environ.get('NEXQUS_TEST_CUDA_DEVICE','cuda:0')


if __name__=='__main__': unittest.main()
