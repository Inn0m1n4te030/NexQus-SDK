"""Train both image QGANs and a classical baseline locally; no QPU calls.

python examples/image_gan.py --mode all --steps 20
python examples/image_gan.py --mode classical --architecture conv --data data/pets --size 64 --channels 3 --epochs 100 --batch-size 64 --hidden 256 --latent-dim 64 --learning-rate 0.0002
python examples/image_gan.py --cifar10 data/cifar10 --download --size 16 --channels 3
See docs/IMAGE_GANS.md for installation, interpretation and resuming.
"""
import argparse
import base64
import html
import json
import math
from pathlib import Path
import platform
from time import perf_counter
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nexqus
from nexqus.image_gan import (QuantumPatchGenerator, HybridImageGenerator,
    ClassicalImageGenerator, ImageGAN, ImageDataset, synthetic_bars,
    load_cifar10_cats_dogs, save_image_grid)
from nexqus.torch_circuit import tensor_device


from nexqus.image_metrics import image_diagnostics, nearest_training_images, hybrid_diagnostics


def make_effect_plot(model, folder, seed):
    device=next(model.parameters()).device
    noise = torch.randn(1, model.latent_dim, generator=torch.Generator(device=device).manual_seed(seed),device=device)
    labels = torch.zeros(1, dtype=torch.long,device=device)
    effect = model.parameter_effect(noise, labels, delta=.25)
    figure, axes = plt.subplots(1, 4, figsize=(11, 3))
    for axis, key, title in zip(axes[:3], ('minus', 'center', 'plus'), ('w − 0.25', 'w', 'w + 0.25')):
        pixels = effect[key][0].cpu().numpy().transpose(1, 2, 0)
        axis.imshow(pixels[:, :, 0] if model.channels == 1 else pixels, cmap='gray', vmin=0, vmax=1,
                    interpolation='nearest'); axis.set_title(title); axis.axis('off')
    sensitivity = effect['sensitivity'][0].mean(0).cpu().numpy()
    limit = max(float(np.max(np.abs(sensitivity))), 1e-9)
    artist = axes[3].imshow(sensitivity, cmap='RdBu_r', vmin=-limit, vmax=limit, interpolation='nearest')
    axes[3].set_title('Pixel change / radian\n(mean over channels)'); axes[3].axis('off')
    figure.colorbar(artist, ax=axes[3], shrink=.65)
    figure.suptitle(effect['parameter']+' · same noise and class'); figure.tight_layout()
    figure.savefig(folder/'parameter_effect.png', dpi=130); plt.close(figure)
    model.circuit_report(noise, labels).to_html(filename=folder/'circuit.html')
    return dict(parameter=effect['parameter'], value=effect['value'], delta=effect['delta'],
                max_abs_pixel_sensitivity=float(effect['sensitivity'].abs().max()))


def write_report(folder, payload):
    def picture(path, caption):
        data = base64.b64encode(path.read_bytes()).decode('ascii')
        return '<figure><img alt="'+html.escape(caption, quote=True)+'" src="data:image/png;base64,'+data+'"><figcaption>'+html.escape(caption)+'</figcaption></figure>'
    def details(title, content):
        return '<details><summary>'+html.escape(title)+'</summary>'+content+'</details>'
    pieces = ['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">',
        '<title>NexQus generated images</title><style>body{font:16px system-ui;max-width:1120px;margin:32px auto;padding:0 20px;background:#101621;color:#e5ecf8}h1,h2{color:#8edcfa}img{max-width:100%;image-rendering:pixelated}figure{margin:20px 0}figcaption,p{line-height:1.6}pre{white-space:pre-wrap;background:#202b3e;padding:16px;border-radius:12px}a{color:#9de4ff}details{margin:16px 0}summary{cursor:pointer}</style>',
        '<h1>Generated images</h1><p>'+html.escape(payload['source'])+'</p>',
        '<p>Rows: '+html.escape(', '.join(payload['classes']))+'. Eight generated samples per class, with fixed noise across progress snapshots. Real training photos are at the bottom.</p>',
        '<p>Inspect recognizable structure, variety and class agreement. GAN losses and pixel statistics do not measure semantic image quality. Software checks do not validate recognizable animal generation.</p>']
    for record in payload['models']:
        mode = record['kind']; directory = folder/mode
        pieces += ['<h2>'+html.escape(mode)+'</h2>', picture(directory/'after.png',
            'GENERATED: after '+str(record['total_steps'])+' total updates; '+record['sampling_weights']+' weights.')]
        for name, stats in record['after'].items():
            fraction = stats['near_duplicate_pair_fraction']
            if fraction is not None:
                pieces += ['<p>'+html.escape(name)+': '+str(round(100*fraction, 1))+'% of displayed pairs have pixel MAE below 0.01. Diverse noise can also score well; this is not a quality score.</p>']
        pieces += [details('Additional independent noise samples', picture(directory/'independent.png',
            'Generated with a separate fixed noise seed; '+str(record['diagnostic_samples_per_class'])+' samples per class.')+
            '<pre>'+html.escape(json.dumps(record['independent_diagnostics'], indent=2))+'</pre>')]
        if 'hybrid_bottleneck' in record:
            pieces += ['<p>Hybrid decoder inputs: '+html.escape(record['hybrid_bottleneck']['decoder_inputs'])+
                '. A classical bypass can dominate image generation; its presence does not establish quantum contribution.</p>']
        pieces += [details('Nearest training references for the displayed outputs', picture(directory/'nearest_training.png',
            'REAL training references, matched to generated grid positions by same-class pooled pixel MAE. This is not a perceptual similarity or memorization test.'))]
        if (directory/'after_raw.png').exists() and record['sampling_weights'] == 'EMA':
            pieces += [details('Current weights without averaging', picture(directory/'after_raw.png', 'Generated using current generator weights with the same fixed noise.'))]
        progress = picture(directory/'before.png', 'Generated before this training session.')+picture(directory/'losses.png', 'Alternating adversarial losses, not image-quality scores.')
        for preview in record.get('progress_previews', []):
            progress += picture(directory/preview['file'], 'Generated at total update '+str(preview['step'])+'.')
        pieces += [details('Training progress', progress)]
        if mode != 'classical':
            pieces += [details('Quantum parameter intervention', picture(directory/'parameter_effect.png',
                'One local parameter intervention on the displayed sampling model. Signed sensitivity, not feature importance.')+
                '<p><a href="'+mode+'/circuit.html">Gate-by-gate circuit explanation</a></p>')]
        pieces += [details('Model settings and diagnostics', '<pre>'+html.escape(json.dumps(record, indent=2))+'</pre>')]
    pieces += ['<h2>Real reference photos</h2>', picture(folder/'training_images.png', 'REAL training examples. These are not generated.'),
        picture(folder/'validation_images.png', 'REAL held-out examples. Not used in gradient updates.'),
        details('Run settings', '<pre>'+html.escape(json.dumps({k:v for k,v in payload.items() if k != 'models'}, indent=2))+'</pre>'), '</html>']
    (folder/'report.html').write_text('\n'.join(pieces), encoding='utf-8')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['all', 'patch', 'hybrid', 'classical'], default='all')
    parser.add_argument('--architecture', choices=['conv', 'mlp', 'dcgan'], default='conv',
                        help='Decoder for hybrid/classical and discriminator for every mode; default conv. Use mlp for old checkpoints.')
    parser.add_argument('--training', choices=['legacy', 'regularized'], default='legacy', help='Opt-in regularized preset; old checkpoint training stays available')
    parser.add_argument('--generator-lr', type=float); parser.add_argument('--discriminator-lr', type=float)
    parser.add_argument('--r1-gamma', type=float); parser.add_argument('--r1-every', type=int, default=16)
    parser.add_argument('--ema-decay', type=float)
    parser.add_argument('--diff-augment', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--readout', choices=['z', 'probabilities']); parser.add_argument('--angle-map', choices=['tanh', 'linear'])
    parser.add_argument('--latent-skip', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--discriminator-hidden', type=int)
    parser.add_argument('--diagnostic-samples', type=int, default=32, help='Independent noise samples per class, at most 128 total')
    parser.add_argument('--retune', action='store_true', help='Explicitly change training controls on resume; preserves learned weights and optimizer moments')
    parser.add_argument('--keep-checkpoints', action='store_true', help='Also save numbered periodic checkpoints')
    parser.add_argument('--overwrite', action='store_true', help='Explicitly allow a fresh run to replace an existing output checkpoint')
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--data', type=Path, help='Root with class subfolders, e.g. cats/ and dogs/')
    source.add_argument('--cifar10', type=Path, help='Optional torchvision CIFAR-10 root')
    parser.add_argument('--download', action='store_true', help='Explicitly allow CIFAR-10 download')
    parser.add_argument('--size', type=int, default=8); parser.add_argument('--channels', type=int, choices=[1,3], default=1)
    parser.add_argument('--patch-size', type=int, choices=[2,4,8])
    parser.add_argument('--depth', type=int, default=1); parser.add_argument('--qubits', type=int, default=4)
    parser.add_argument('--latent-dim', type=int, default=4); parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--samples', type=int, default=64)
    budget = parser.add_mutually_exclusive_group()
    budget.add_argument('--steps', type=int, help='Alternating updates in this session (default 20 if no budget is given)')
    budget.add_argument('--epochs', type=int, help='Passes through training data in this session; finishes a partial epoch on resume')
    parser.add_argument('--batch-size', type=int, default=4); parser.add_argument('--learning-rate', type=float, default=.002)
    parser.add_argument('--seed', type=int, default=7); parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--data-seed', type=int, help='Seeded per-class file selection instead of sorted truncation; keep unchanged on resume')
    parser.add_argument('--horizontal-flip', action='store_true', help='Random training-only horizontal flips; RNG saved in checkpoint')
    parser.add_argument('--preview-every', type=int, default=500, help='Save fixed-noise progress PNG every N session updates; 0 disables')
    parser.add_argument('--checkpoint-every', type=int, default=1000, help='Atomically refresh checkpoint every N session updates; 0 disables')
    parser.add_argument('--log-every', type=int, default=100)
    parser.add_argument('--resume', type=Path, help='Resume one mode with identical data/split/batch size')
    parser.add_argument('--device',default=None,help='cpu, cuda, cuda:N or auto; otherwise NEXQUS_DEVICE or cpu')
    parser.add_argument('--reset-rng',action='store_true',help='Allow cross CPU/CUDA checkpoint continuation with a new noise stream')
    parser.add_argument('--output-dir', type=Path, default=Path('.release-artifacts/image-gans'))
    args = parser.parse_args(argv)
    if not 1 <= args.batch_size <= 128 or not 1 <= args.diagnostic_samples <= 128:
        parser.error('Batch size and diagnostic samples must be 1..128')
    if args.retune and not args.resume: parser.error('--retune requires --resume')
    regularized = args.training == 'regularized'
    options = dict(training=args.training,
        generator_learning_rate=args.learning_rate if args.generator_lr is None else args.generator_lr,
        discriminator_learning_rate=(args.learning_rate*(.5 if regularized else 1) if args.discriminator_lr is None else args.discriminator_lr),
        r1_gamma=(1. if regularized else 0.) if args.r1_gamma is None else args.r1_gamma,
        r1_every=args.r1_every, diff_augment=regularized if args.diff_augment is None else args.diff_augment,
        ema_decay=(.995 if regularized else 0.) if args.ema_decay is None else args.ema_decay)
    d_hidden = args.hidden if args.discriminator_hidden is None else args.discriminator_hidden
    device=tensor_device(args.device)
    print('Execution device:',device,torch.cuda.get_device_name(device) if device.type=='cuda' else '',flush=True)
    if args.download and args.cifar10 is None: parser.error('--download requires --cifar10')
    if args.steps is None and args.epochs is None: args.steps = 20
    if (any(v is not None and v < 1 for v in (args.steps, args.epochs))
            or args.threads < 1 or args.log_every < 1 or args.preview_every < 0 or args.checkpoint_every < 0):
        parser.error('Budgets, threads and log interval must be positive; preview/checkpoint intervals must be nonnegative')
    if args.resume and args.mode == 'all': parser.error('--resume requires one specific --mode')
    torch.set_num_threads(args.threads)
    if args.data:
        data = ImageDataset.from_folder(args.data, image_size=args.size, channels=args.channels,
                                        max_images=args.samples, selection_seed=args.data_seed)
    elif args.cifar10:
        data = load_cifar10_cats_dogs(args.cifar10, image_size=args.size, channels=args.channels,
                                     max_images=args.samples, download=args.download)
    else:
        data = synthetic_bars(samples=args.samples, image_size=args.size, channels=args.channels, seed=args.seed)
    training, validation = data.split(seed=args.seed)
    modes = ['patch', 'hybrid', 'classical'] if args.mode == 'all' else [args.mode]
    folder = args.output_dir
    if not args.resume and not args.overwrite and any((folder/mode/'checkpoint.pt').exists() for mode in modes):
        parser.error('Output already has a checkpoint; use --resume, a new --output-dir, or explicit --overwrite')
    folder.mkdir(parents=True, exist_ok=True)
    for subset, name in ((training, 'training'), (validation, 'validation')):
        real_rows = []
        for i in range(len(data.class_names)):
            images = subset.images[subset.labels == i]
            real_rows.append(images[torch.arange(8) % len(images)])
        save_image_grid(torch.cat(real_rows), folder/(name+'_images.png'), columns=8)
    steps_per_epoch = math.ceil(len(training)/args.batch_size)
    print(f'{len(training)} training / {len(validation)} validation images; {steps_per_epoch} updates per epoch.', flush=True)
    labels = torch.arange(len(data.class_names)).repeat_interleave(8)
    if len(labels) > 128: parser.error('The comparison preview supports at most 16 classes')
    payload = dict(sdk_version=nexqus.__version__, torch_version=str(torch.__version__), numpy_version=np.__version__,
        platform=platform.platform(), device=str(device),requested_device=args.device,
        gpu_name=torch.cuda.get_device_name(device) if device.type=='cuda' else None,
        precision='float32 Torch / batched complex64 Torch circuit autograd',
        threads=torch.get_num_threads(), source=data.source, classes=list(data.class_names),
        shape=list(data.images.shape[1:]), training_samples=len(training), validation_samples=len(validation),
        training_fingerprint=training.fingerprint, validation_fingerprint=validation.fingerprint,
        seed=args.seed, batch_size=args.batch_size, learning_rate=args.learning_rate,
        requested_session_steps=args.steps, requested_session_epochs=args.epochs, updates_per_epoch=steps_per_epoch,
        training_options=options, retuned=args.retune, architecture=args.architecture, data_selection_seed=args.data_seed, horizontal_flip=args.horizontal_flip,
        comparison='Matched updates/data/D initialization within one run; generator parameter budgets differ',
        image_quality=dict(status='not_evaluated', software_checks_are_not_semantic_validation=True),
        diagnostic_definition='Pixel mean, spread and within-class pairwise/duplicate statistics; neither semantic quality nor fidelity',
        models=[])
    for mode in modes:
        output = folder/mode; output.mkdir(parents=True, exist_ok=True)
        common = dict(image_size=args.size, channels=args.channels, latent_dim=args.latent_dim,
                      num_classes=len(data.class_names), seed=args.seed)
        if args.resume:
            trainer = ImageGAN.load(args.resume,device=device,reset_rng=args.reset_rng)
            if trainer.generator.config['kind'] != mode: parser.error('Checkpoint kind differs from --mode')
            if trainer.config['seed'] != args.seed: parser.error('Keep --seed identical to checkpoint for a matched continuation')
            g = trainer.generator
            if any(g.config[key] != value for key, value in common.items()):
                parser.error('Keep image size, channels, latent dimension, class count and seed identical to the checkpoint')
            if trainer.config['discriminator_hidden'] != d_hidden or (mode != 'patch' and g.config['hidden'] != args.hidden):
                parser.error('Keep generator/discriminator hidden sizes identical to the checkpoint')
            if args.retune: trainer.retune(**options)
            elif any(trainer.config[k] != v for k,v in options.items()):
                parser.error('Keep training controls identical or explicitly use --retune')
            if mode == 'hybrid' and any(value is not None and g.config[key] != value
                    for key, value in (('readout', args.readout), ('angle_map', args.angle_map), ('latent_skip', args.latent_skip))):
                parser.error('Readout, angle map and latent skip are architecture settings; start a new model to change them')
            if (g.config.get('architecture', args.architecture) != args.architecture
                    or trainer.config['discriminator_architecture'] != args.architecture
                    or trainer.config['horizontal_flip'] != args.horizontal_flip):
                parser.error('Keep architecture and horizontal-flip setting identical to the checkpoint')
            if mode == 'hybrid' and (g.config['n_qubits'] != args.qubits or g.config['depth'] != args.depth):
                parser.error('Keep qubits and depth identical to the checkpoint')
            if mode == 'patch' and (g.config['depth'] != args.depth
                    or g.config['patch_size'] != (args.patch_size or (8 if args.size >= 32 else 4))):
                parser.error('Keep patch size and depth identical to the checkpoint')
        else:
            if mode == 'patch':
                g = QuantumPatchGenerator(**common, depth=args.depth, patch_size=args.patch_size or (8 if args.size >= 32 else 4))
            elif mode == 'hybrid': g = HybridImageGenerator(**common, n_qubits=args.qubits, depth=args.depth,
                                                           hidden=args.hidden, architecture=args.architecture,
                    readout=args.readout or ('probabilities' if regularized else 'z'),
                    angle_map=args.angle_map or ('linear' if regularized else 'tanh'), latent_skip=bool(args.latent_skip))
            else: g = ClassicalImageGenerator(**common, hidden=args.hidden, architecture=args.architecture)
            trainer = ImageGAN(g, learning_rate=args.learning_rate, discriminator_hidden=d_hidden, seed=args.seed,
                device=device, discriminator_architecture=args.architecture, horizontal_flip=args.horizontal_flip, **options)
        before = trainer.sample(len(labels), labels=labels, seed=1000+args.seed)
        save_image_grid(before, output/'before.png', columns=8)
        start_steps = trainer.steps
        progress_previews = []; export_seconds = 0.
        def progress(record):
            nonlocal export_seconds
            session_step = record['step']-start_steps
            if session_step == 1 or session_step % args.log_every == 0:
                print(f"{mode}: step {record['step']} epoch {trainer.epoch+trainer._cursor/len(training):.2f}  D={record['discriminator_loss']:.4f} G={record['generator_loss']:.4f}", flush=True)
            started_export = perf_counter()
            if args.preview_every and session_step % args.preview_every == 0:
                filename = f'progress/step_{record["step"]:08d}.png'
                save_image_grid(trainer.sample(len(labels), labels=labels, seed=1000+args.seed), output/filename, columns=8)
                progress_previews.append(dict(step=record['step'], file=filename))
            if args.checkpoint_every and session_step % args.checkpoint_every == 0:
                trainer.save(output/'checkpoint.pt')
                if args.keep_checkpoints: trainer.save(output/f'checkpoints/step_{record["step"]:08d}.pt')
            export_seconds += perf_counter()-started_export
        if device.type=='cuda': torch.cuda.synchronize(device)
        started = perf_counter()
        trainer.fit(training, epochs=args.epochs or math.ceil(args.steps/steps_per_epoch)+1,
                    batch_size=args.batch_size, max_steps=args.steps, callback=progress)
        if device.type=='cuda': torch.cuda.synchronize(device)
        elapsed = perf_counter()-started-export_seconds
        after = trainer.sample(len(labels), labels=labels, seed=1000+args.seed)
        save_image_grid(after, output/'after.png', columns=8); trainer.save(output/'checkpoint.pt')
        if trainer.ema_generator is not None:
            save_image_grid(trainer.sample(len(labels), labels=labels, seed=1000+args.seed, use_ema=False), output/'after_raw.png', columns=8)
        diagnostic_count = min(args.diagnostic_samples, 128//len(data.class_names))
        diagnostic_labels = torch.arange(len(data.class_names)).repeat_interleave(diagnostic_count)
        independent = trainer.sample(len(diagnostic_labels), labels=diagnostic_labels, seed=2000+args.seed)
        save_image_grid(independent, output/'independent.png', columns=min(8, diagnostic_count))
        nearest, nearest_stats = nearest_training_images(after, labels, training)
        save_image_grid(nearest, output/'nearest_training.png', columns=8)
        figure, axis = plt.subplots(figsize=(6,3))
        for key in ('discriminator_loss', 'generator_loss'):
            axis.plot([row['step'] for row in trainer.history], [row[key] for row in trainer.history], label=key)
        axis.set(xlabel='Alternating update', ylabel='Loss', title=mode+' training'); axis.legend(); figure.tight_layout()
        figure.savefig(output/'losses.png', dpi=130); plt.close(figure)
        record = dict(kind=mode, generator_config=g.config, trainer_config=trainer.config,
            sampling_weights='EMA' if trainer.ema_generator is not None else 'current',
            diagnostic_samples_per_class=diagnostic_count, diagnostic_seed=2000+args.seed,
            independent_diagnostics=image_diagnostics(independent, diagnostic_labels, validation), nearest_training=nearest_stats,
            session_steps=trainer.steps-start_steps, total_steps=trainer.steps,
            completed_epochs=trainer.epoch, epoch_fraction=trainer._cursor/len(training),
            progress_previews=progress_previews, periodic_export_seconds=export_seconds,
            generator_parameters=sum(p.numel() for p in g.parameters()),
            quantum_parameters=sum(q.weights.numel() for q in g.quantum_layers()),
            discriminator_parameters=sum(p.numel() for p in trainer.discriminator.parameters()),
            session_training_seconds=elapsed, timing_scope='Training loop and logging including model forward/backward and optimizers; CUDA synchronized; excludes setup, periodic preview/checkpoint export and final export',
            before=image_diagnostics(before, labels, validation), after=image_diagnostics(after, labels, validation),
            final_losses=trainer.history[-1])
        sampling_model = trainer.sampling_generator()
        if mode == 'hybrid':
            probe = torch.randn(len(diagnostic_labels), g.latent_dim, generator=torch.Generator(device=device).manual_seed(args.seed+1009), device=device)
            record['hybrid_bottleneck'] = hybrid_diagnostics(sampling_model, probe, diagnostic_labels.to(device))
        if mode != 'classical': record['parameter_effect'] = make_effect_plot(sampling_model, output, args.seed)
        payload['models'].append(record)
        print('Generated images:', (output/'after.png').resolve(), flush=True)
        print(f'{mode}: saved checkpoint and previews ({elapsed:.2f}s training)', flush=True)
    (folder/'metrics.json').write_text(json.dumps(payload, indent=2)+'\n', encoding='utf-8')
    write_report(folder, payload)
    print('Open', (folder/'report.html').resolve())
    return payload


if __name__ == '__main__': main()
