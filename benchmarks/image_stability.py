"""Bounded real-image GAN trials; no automatic downloads or QPU calls.

python benchmarks/image_stability.py --data data/pets --steps 1000
Alternatively install pyarrow separately and pass --cifar-parquet FILE from the
UofT CIFAR mirror. Neither pyarrow nor the dataset becomes an SDK dependency.
"""
import argparse
import base64
from hashlib import sha256
import html
import io
import json
from pathlib import Path
import platform
import sys
from time import perf_counter
import numpy as np
import torch
from PIL import Image
from nexqus.image_gan import ImageDataset, ImageGAN, ClassicalImageGenerator, HybridImageGenerator, save_image_grid
from nexqus.image_metrics import image_diagnostics, hybrid_diagnostics
from nexqus.torch_circuit import tensor_device

MIRROR = 'https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/train-00000-of-00001.parquet'
CASES = ['classical-legacy', 'classical-regularized', 'hybrid-legacy', 'hybrid-regularized', 'classical-dcgan']


def load_data(args):
    if args.data:
        return ImageDataset.from_folder(args.data, image_size=args.size, channels=3,
            max_images=args.samples, selection_seed=args.seed), dict(kind='local folders', path=str(args.data))
    import pyarrow.parquet as pq
    rows = pq.read_table(args.cifar_parquet).to_pydict(); rng = np.random.default_rng(args.seed)
    images = []; labels = []
    for label, cifar in enumerate((3,5)):
        selected = rng.permutation(np.flatnonzero(np.asarray(rows['label']) == cifar))[:args.samples//2]
        for index in selected:
            with Image.open(io.BytesIO(rows['img'][int(index)]['bytes'])) as image:
                image = image.convert('RGB').resize((args.size,args.size), Image.Resampling.BILINEAR)
                images.append(torch.from_numpy(np.array(image, copy=True)).permute(2,0,1).float()/255)
                labels.append(label)
    return ImageDataset(torch.stack(images), torch.tensor(labels), ('cat','dog'), source='UofT CIFAR-10 train cat/dog subset'), dict(
        kind='UofT CIFAR-10 train mirror', url=MIRROR, file_sha256=sha256(args.cifar_parquet.read_bytes()).hexdigest())


def write_report(folder, payload):
    pieces = ['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">',
        '<title>Bounded image GAN trials</title><style>body{font:16px system-ui;max-width:1100px;margin:32px auto;padding:16px;background:#101621;color:#e5ecf8}img{max-width:100%;image-rendering:pixelated}pre{white-space:pre-wrap}p{line-height:1.5}</style>',
        '<h1>Actual generated outputs from bounded trials</h1>',
        '<p>Each grid shows 32 samples per class, fixed noise seed 101. First four rows: '+html.escape(payload['classes'][0])+'. Last four rows: '+html.escape(payload['classes'][1])+'. No training photographs are embedded.</p>',
        '<p>Inspect structure, variety and class agreement. Losses and pixel diversity do not establish semantic quality. Zero duplicate pairs can coexist with poor images. A short single-seed pilot cannot validate a pet-image recipe or quantum advantage.</p>']
    if payload.get('observed_quality'): pieces += ['<p><strong>'+html.escape(payload['observed_quality'])+'</strong></p>']
    for record in payload['models']:
        encoded = base64.b64encode((folder/record['preview_path']).read_bytes()).decode('ascii')
        pieces += ['<h2>'+html.escape(record['case'])+'</h2>',
            '<p>'+str(record['steps'])+' updates; '+record['sampling_weights']+' sampling weights.</p>',
            '<img alt="Actual generated trial outputs" src="data:image/png;base64,'+encoded+'">']
    pieces += ['<details><summary>Settings, provenance, limitations and diagnostics</summary><pre>'+html.escape(json.dumps(payload,indent=2))+'</pre></details></html>']
    (folder/'report.html').write_text('\n'.join(pieces),encoding='utf-8')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--data', type=Path); source.add_argument('--cifar-parquet', type=Path)
    parser.add_argument('--steps', type=int, default=1000); parser.add_argument('--samples', type=int, default=2000)
    parser.add_argument('--size', type=int, default=32); parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--seed', type=int, default=7); parser.add_argument('--device', default='cpu')
    parser.add_argument('--cases', nargs='+', choices=CASES, default=CASES)
    parser.add_argument('--output', type=Path, default=Path('.release-artifacts/image-stability'))
    args = parser.parse_args(argv)
    if args.steps < 1 or args.samples < 4 or args.samples > 10000 or args.samples % 2:
        parser.error('Use positive steps and an even sample count 4..10000')
    torch.set_num_threads(1); device = tensor_device(args.device)
    if (args.output/'metrics.json').exists(): parser.error('Use a fresh output directory to preserve evidence')
    args.output.mkdir(parents=True, exist_ok=True)
    data, provenance = load_data(args)
    if len(data.class_names) != 2: parser.error('This bounded comparison expects two classes')
    training, validation = data.split(seed=args.seed); labels = torch.arange(2).repeat_interleave(32)
    payload = dict(software=dict(torch=str(torch.__version__), numpy=np.__version__, python=sys.version, platform=platform.platform()),
        device=str(device), threads=1, source=provenance, seed=args.seed, batch_size=32, latent_dim=16, classes=list(data.class_names),
        training_samples=len(training), validation_samples=len(validation), training_fingerprint=training.fingerprint,
        validation_fingerprint=validation.fingerprint, image_shape=list(data.images.shape[1:]), evaluation_seed=101,
        evaluation_samples_per_class=32, semantic_quality='not automatically evaluated',
        comparison='Same data/order/prior/updates; model parameter counts differ. Regularized hybrid changes readout, angle map and bypass together, not an ablation.',
        limitations=['Single seed and bounded steps; not validation of the 64x64 Kaggle pet workflow.',
            'Training times are single-session records, not comparative performance benchmarks.',
            'Pixel diversity is not semantic quality; no quantum advantage claim.'], models=[])
    for case in args.cases:
        folder = args.output/case; folder.mkdir(); mode, variant = case.split('-',1)
        regularized = variant != 'legacy'; architecture = 'dcgan' if variant == 'dcgan' else 'conv'
        common = dict(image_size=args.size, channels=3, latent_dim=16, hidden=args.hidden, num_classes=2, seed=args.seed, architecture=architecture)
        generator = (HybridImageGenerator(**common, readout='probabilities' if regularized else 'z',
            angle_map='linear' if regularized else 'tanh', latent_skip=regularized)
            if mode == 'hybrid' else ClassicalImageGenerator(**common))
        trainer = ImageGAN(generator, device=device, learning_rate=.0002, discriminator_hidden=args.hidden, seed=args.seed,
            horizontal_flip=True, training='regularized' if regularized else 'legacy',
            generator_learning_rate=.0002, discriminator_learning_rate=.0001 if regularized else .0002,
            r1_gamma=1. if regularized else 0., r1_every=16, diff_augment=regularized, ema_decay=.995 if regularized else 0.)
        before = trainer.sample(64, labels=labels, seed=101); save_image_grid(before, folder/'before.png', columns=8, scale=2)
        def progress(record):
            if record['step'] % 250 == 0: print(case, record['step'], flush=True)
        if device.type == 'cuda': torch.cuda.synchronize(device)
        started = perf_counter()
        trainer.fit(training, epochs=args.steps+1, batch_size=32, max_steps=args.steps, callback=progress)
        if device.type == 'cuda': torch.cuda.synchronize(device)
        elapsed = perf_counter()-started
        after = trainer.sample(64, labels=labels, seed=101); raw = trainer.sample(64, labels=labels, seed=101, use_ema=False)
        save_image_grid(after, folder/'after.png', columns=8, scale=2); save_image_grid(raw, folder/'raw.png', columns=8, scale=2); trainer.save(folder/'checkpoint.pt')
        record = dict(case=case, generator_config=generator.config, trainer_config=trainer.config, steps=trainer.steps,
            generator_parameters=sum(p.numel() for p in generator.parameters()), discriminator_parameters=sum(p.numel() for p in trainer.discriminator.parameters()),
            sampling_weights='EMA' if regularized else 'current', training_seconds=elapsed,
            timing_scope='Training and logging, synchronized CUDA; excludes setup, sampling and export. Single session, no repeated timing benchmark.',
            before=image_diagnostics(before,labels,validation), after=image_diagnostics(after,labels,validation), raw=image_diagnostics(raw,labels,validation),
            final_losses=trainer.history[-1], preview_path=case+'/after.png')
        if mode == 'hybrid':
            noise = torch.randn(64,16,generator=torch.Generator(device=device).manual_seed(101),device=device)
            record['hybrid_bottleneck'] = hybrid_diagnostics(trainer.sampling_generator(),noise,labels.to(device))
        payload['models'].append(record)
        (args.output/'metrics.json').write_text(json.dumps(payload,indent=2)+'\n',encoding='utf-8'); write_report(args.output,payload)
    print(args.output/'report.html',flush=True)


if __name__ == '__main__': main()
