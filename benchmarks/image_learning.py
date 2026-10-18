"""Reproduce the bounded convolutional bars experiment; no dataset download/QPU.

This measures requested line orientation, not pet semantics, fidelity or diversity.
"""
import argparse
import json, torch
from pathlib import Path
from nexqus.image_gan import ClassicalImageGenerator, HybridImageGenerator, ImageGAN, synthetic_bars, save_image_grid
torch.set_num_threads(1)
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,default=Path('.release-artifacts/conv-learning'))
args=parser.parse_args()
folder=args.output; folder.mkdir(parents=True,exist_ok=True)
data=synthetic_bars(samples=128,image_size=16,seed=29)
save_image_grid(data.images[:64],folder/'training.png',columns=8)
results=[]
for constructor in (ClassicalImageGenerator, HybridImageGenerator):
    model=constructor(image_size=16,latent_dim=8,hidden=32,architecture='conv',seed=7)
    trainer=ImageGAN(model,learning_rate=.0002,discriminator_hidden=32,seed=7)
    labels=torch.arange(2).repeat_interleave(32)
    def stats(images):
        gray=images.mean(1)
        vertical=gray.mean(1).var(1,unbiased=False)
        horizontal=gray.mean(2).var(1,unbiased=False)
        correct=torch.where(labels==0,vertical>horizontal,horizontal>vertical)
        return dict(orientation_accuracy=float(correct.float().mean()),mean=float(images.mean()),pixel_std=float(images.std()),orientation_contrast=float((vertical-horizontal).abs().mean()))
    before=trainer.sample(64,labels=labels,seed=101)
    save_image_grid(before,folder/(model.config['kind']+'-before.png'),columns=8)
    record=dict(kind=model.config['kind'],before=stats(before),snapshots=[])
    def callback(row):
        if row['step']%200==0:
            sample=trainer.sample(64,labels=labels,seed=101)
            metrics=dict(step=row['step'],**stats(sample))
            print(model.config['kind'],metrics,flush=True)
            record['snapshots'].append(metrics)
    trainer.fit(data,epochs=125,batch_size=16,max_steps=1000,callback=callback)
    after=trainer.sample(64,labels=labels,seed=101)
    save_image_grid(after,folder/(model.config['kind']+'-after.png'),columns=8)
    record['after']=stats(after); results.append(record)
    (folder/'metrics.json').write_text(json.dumps(dict(
        scope='Single-seed procedural bars; not pet-image validation or a quantum-advantage benchmark',
        torch_version=str(torch.__version__),device='cpu',threads=1,steps=1000,batch_size=16,
        samples=128,image_size=16,channels=1,latent_dim=8,hidden=32,learning_rate=.0002,
        seed=7,data_seed=29,evaluation_seed=101,evaluation_samples=64,
        orientation_definition='Vertical if variance of column means exceeds variance of row means, otherwise horizontal; ties count incorrect',
        limitation='Orientation alone does not establish realistic bars, location diversity, generalization or animal semantics; inspect previews',
        models=results),indent=2))
print(json.dumps(results,indent=2))
