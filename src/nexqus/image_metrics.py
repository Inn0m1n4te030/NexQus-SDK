"""Pixel diversity and latent bottleneck diagnostics, not semantic quality scores."""
import math
import torch
from torch.nn import functional as F
from .image_data import ImageDataset


def _inputs(images, labels):
    if (not isinstance(images, torch.Tensor) or not images.is_floating_point()
            or images.ndim != 4 or not 1 <= len(images) <= 128
            or not torch.isfinite(images).all() or torch.any(images < 0) or torch.any(images > 1)):
        raise ValueError('Expected 1..128 finite NCHW images in [0,1]')
    if not isinstance(labels, torch.Tensor) or labels.dtype != torch.long or labels.shape != (len(images),):
        raise ValueError('Expected one int64 label per image')
    return images.detach().cpu(), labels.detach().cpu()


def _reference(images, labels, reference):
    if not isinstance(reference, ImageDataset) or images.shape[1:] != reference.images.shape[1:]:
        raise ValueError('Reference dataset shape differs')
    if torch.any(labels < 0) or torch.any(labels >= len(reference.class_names)):
        raise ValueError('Invalid class label')


def _pooled(images):
    return F.adaptive_avg_pool2d(images, (min(8, images.shape[-2]), min(8, images.shape[-1]))).flatten(1)


def image_diagnostics(generated, labels, reference):
    generated, labels = _inputs(generated, labels); _reference(generated, labels, reference)
    result = {}
    for label, name in enumerate(reference.class_names):
        images = generated[labels == label]
        if not len(images): continue
        fake = images.flatten(1); real = reference.images[reference.labels == label].flatten(1)
        pairs = torch.pdist(fake, p=1)/fake.shape[1]
        pooled = _pooled(images); pooled_pairs = torch.pdist(pooled, p=1)/pooled.shape[1]
        result[name] = dict(sample_count=len(fake),
            class_mean_pixel_mae=float((fake.mean(0)-real.mean(0)).abs().mean()),
            generated_pixel_std=float(fake.std(0, unbiased=False).mean()), pair_count=len(pairs),
            mean_pairwise_pixel_mae=float(pairs.mean()) if len(pairs) else None,
            closest_pair_pixel_mae=float(pairs.min()) if len(pairs) else None,
            near_duplicate_pair_fraction=float((pairs < .01).float().mean()) if len(pairs) else None,
            near_duplicate_threshold=.01,
            mean_pairwise_pooled_mae=float(pooled_pairs.mean()) if len(pooled_pairs) else None)
    return result


def nearest_training_images(generated, labels, training):
    """Same-class nearest 8x8 pooled pixel L1 reference; not a memorization test."""
    generated, labels = _inputs(generated, labels); _reference(generated, labels, training)
    matches = []; indices = []; distances = []
    for image, label in zip(generated, labels):
        candidates = torch.where(training.labels == label)[0]
        real = _pooled(training.images[candidates]); query = _pooled(image[None])
        distance = torch.cdist(query, real, p=1)[0]/real.shape[1]
        closest = int(distance.argmin()); index = int(candidates[closest])
        matches.append(training.images[index]); indices.append(index); distances.append(float(distance[closest]))
    return torch.stack(matches).clone(), dict(metric='same-class pooled 8x8 pixel MAE', indices=indices, distances=distances)


def hybrid_diagnostics(model, noise, labels):
    model._inputs(noise, labels); previous = model.training; model.eval()
    try:
        with torch.no_grad():
            angles = model.encoded_angles(noise, labels); features = model.quantum_features(noise, labels)
            classes = {}
            for label in labels.unique().tolist():
                mask = labels == label; selected = angles[mask]
                classes[str(label)] = dict(sample_count=int(mask.sum()),
                    mean_angle_std=float(selected.std(0, unbiased=False).mean()),
                    mean_quantum_feature_std=float(features[mask].std(0, unbiased=False).mean()),
                    tanh_saturation_fraction=float((selected.abs() > .95*math.pi).float().mean())
                        if model.config['angle_map'] == 'tanh' else None)
            return dict(readout=model.config['readout'], angle_map=model.config['angle_map'],
                latent_skip=model.config['latent_skip'], quantum_feature_dimension=features.shape[1],
                decoder_inputs='quantum features plus classical noise' if model.config['latent_skip'] else 'quantum features only',
                classes=classes)
    finally: model.train(previous)
