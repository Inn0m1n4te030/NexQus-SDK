"""Local image datasets and explicit optional CIFAR-10 cat/dog loading."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageOps
from .image_models import _image_shape
from .circuit import positive_integer


def _pixels(image, image_size, channels):
    image = ImageOps.exif_transpose(image).convert('L' if channels == 1 else 'RGB')
    image = ImageOps.fit(image, (image_size, image_size), method=Image.Resampling.BILINEAR)
    array = np.array(image, dtype=np.float32, copy=True)/255
    if channels == 1: array = array[:, :, None]
    return torch.from_numpy(array.transpose(2, 0, 1).copy())


class ImageDataset:
    """Bounded in-memory CPU image data in [0,1], with explicit class names."""
    def __init__(self, images, labels, class_names, *, source='Caller-supplied image tensors'):
        class_names = tuple(class_names)
        if (not 1 <= len(class_names) <= 32
                or any(not isinstance(s, str) or not s or len(s) > 160 for s in class_names)
                or len(set(class_names)) != len(class_names)):
            raise ValueError('Provide 1..32 distinct class names')
        if not isinstance(images, torch.Tensor) or images.device.type != 'cpu' or images.is_complex():
            raise ValueError('Images must be real CPU tensors')
        if (images.ndim != 4 or not 1 <= len(images) <= 10000 or images.shape[2] != images.shape[3]
                or not torch.isfinite(images).all() or images.min() < 0 or images.max() > 1):
            raise ValueError('Images must have shape (N=1..10000, C, size, size) with finite pixels in [0,1]')
        _image_shape(images.shape[2], images.shape[1])
        if (not isinstance(labels, torch.Tensor) or labels.device.type != 'cpu' or labels.dtype != torch.long
                or labels.shape != (len(images),) or torch.any(labels < 0) or torch.any(labels >= len(class_names))):
            raise ValueError('Provide valid CPU int64 class labels')
        if set(labels.tolist()) != set(range(len(class_names))): raise ValueError('Every declared class needs data')
        self.images = images.detach().to(torch.float32).contiguous().clone()
        self.labels = labels.detach().clone(); self.class_names = class_names; self.source = str(source)

    def __len__(self): return len(self.images)

    @property
    def fingerprint(self):
        h = hashlib.sha256(json.dumps(self.class_names).encode())
        h.update(self.images.numpy().tobytes()); h.update(self.labels.numpy().tobytes())
        return h.hexdigest()

    def split(self, validation_fraction=.2, seed=7):
        if not 0 < validation_fraction < 1: raise ValueError('validation_fraction must be between 0 and 1')
        rng = torch.Generator().manual_seed(seed); train = []; validation = []
        for label in range(len(self.class_names)):
            indices = torch.nonzero(self.labels == label).flatten()
            if len(indices) < 2: raise ValueError('Each class needs at least two samples for a disjoint split')
            indices = indices[torch.randperm(len(indices), generator=rng)]
            count = max(1, min(len(indices)-1, round(len(indices)*validation_fraction)))
            validation.extend(indices[:count].tolist()); train.extend(indices[count:].tolist())
        return (ImageDataset(self.images[train], self.labels[train], self.class_names, source=self.source+' / train'),
                ImageDataset(self.images[validation], self.labels[validation], self.class_names, source=self.source+' / validation'))

    @classmethod
    def from_folder(cls, root, image_size=16, channels=3, max_images=256, selection_seed=None):
        """Read root/class_name/*.(jpg,png,...) without network access.

        Files are capped equally per class (sorted, or shuffled with selection_seed), then center-cropped,
        resized and mapped to [0,1]. No augmentation or held-out leakage.
        """
        image_size, channels = _image_shape(image_size, channels)
        max_images = positive_integer(max_images, 'max_images')
        if max_images > 10000: raise ValueError('max_images must be at most 10000')
        root = Path(root); extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        if not root.is_dir(): raise ValueError('Image root must be a directory with one subdirectory per class')
        groups = [(p.name, sorted(f for f in p.rglob('*') if f.is_file() and f.suffix.lower() in extensions))
                  for p in sorted(root.iterdir()) if p.is_dir()]
        groups = [(name, files) for name, files in groups if files]
        if not 1 <= len(groups) <= 32 or max_images < len(groups): raise ValueError('No usable classes or insufficient image limit')
        if selection_seed is not None:
            from .image_models import _seed
            rng = torch.Generator().manual_seed(_seed(selection_seed))
            groups = [(name, [files[i] for i in torch.randperm(len(files), generator=rng).tolist()])
                      for name, files in groups]
        quota = max_images//len(groups); images = []; labels = []
        for label, (_, files) in enumerate(groups):
            for file in files[:quota]:
                try:
                    with Image.open(file) as image: images.append(_pixels(image, image_size, channels))
                except (OSError, ValueError) as exc:
                    raise ValueError(f'Cannot read image {file.name}') from exc
                labels.append(label)
        return cls(torch.stack(images), torch.tensor(labels, dtype=torch.long),
                   [name for name, _ in groups], source='Local image folders; center crop + bilinear resize')


def load_cifar10_cats_dogs(root, *, image_size=16, channels=3, max_images=256, train=True, download=False):
    """Explicit optional dataset download via torchvision; no pretrained models.

    CIFAR-10 class IDs 3 (cat), 5 (dog) are mapped to labels 0/1. train=False
    selects the official test set. Download is off unless explicitly enabled.
    """
    image_size, channels = _image_shape(image_size, channels)
    max_images = positive_integer(max_images, 'max_images')
    if not 2 <= max_images <= 10000: raise ValueError('max_images must be 2..10000')
    try:
        from torchvision.datasets import CIFAR10
    except ImportError as exc:
        raise ImportError('Install torchvision compatible with your PyTorch version for optional CIFAR-10 loading') from exc
    dataset = CIFAR10(root=str(root), train=bool(train), download=bool(download))
    images = []; labels = []
    for original, mapped in ((3, 0), (5, 1)):
        count = 0
        for i, target in enumerate(dataset.targets):
            if target == original:
                image, _ = dataset[i]; images.append(_pixels(image, image_size, channels)); labels.append(mapped)
                count += 1
                if count >= max_images//2: break
    return ImageDataset(torch.stack(images), torch.tensor(labels, dtype=torch.long), ('cat', 'dog'),
                        source='CIFAR-10 official '+('training' if train else 'test')+' split / cats and dogs')


def synthetic_bars(samples=64, image_size=8, channels=1, seed=7):
    """Procedural horizontal/vertical bars for offline tests, not animal images."""
    samples = positive_integer(samples, 'samples'); image_size, channels = _image_shape(image_size, channels)
    if not 4 <= samples <= 10000: raise ValueError('Provide 4..10000 samples')
    rng = torch.Generator().manual_seed(seed)
    images = torch.zeros(samples, channels, image_size, image_size); labels = torch.arange(samples) % 2
    for i, label in enumerate(labels):
        position = int(torch.randint(1, image_size-1, (1,), generator=rng))
        width = 1 if image_size < 12 else 2
        if int(label) == 0: images[i, :, :, position:position+width] = 1
        else: images[i, :, position:position+width, :] = 1
        images[i] = (images[i]+.025*torch.rand(images[i].shape, generator=rng)).clamp(0, 1)
    return ImageDataset(images, labels.long(), ('vertical bar', 'horizontal bar'), source='Procedural bars; no real cat/dog data')


def save_image_grid(images, filename, columns=8, scale=8):
    """Write a PNG grid of actual [0,1] grayscale/RGB generator outputs."""
    columns = positive_integer(columns, 'columns'); scale = positive_integer(scale, 'scale')
    if (not isinstance(images, torch.Tensor) or images.ndim != 4 or not len(images)
            or images.shape[1] not in (1, 3) or not torch.isfinite(images).all()
            or images.min() < 0 or images.max() > 1): raise ValueError('Provide finite image tensors in [0,1]')
    data = images.detach().cpu().numpy(); rows = (len(data)+columns-1)//columns
    c, h, w = data.shape[1:]; canvas = np.full((rows*(h+1)+1, columns*(w+1)+1, c), .12, dtype=float)
    for i, image in enumerate(data):
        r, col = divmod(i, columns); canvas[1+r*(h+1):1+r*(h+1)+h, 1+col*(w+1):1+col*(w+1)+w] = image.transpose(1,2,0)
    pixels = np.rint(canvas*255).astype(np.uint8)
    image = Image.fromarray(pixels[:, :, 0] if c == 1 else pixels)
    filename = Path(filename); filename.parent.mkdir(parents=True, exist_ok=True)
    image.resize((image.width*scale, image.height*scale), Image.Resampling.NEAREST).save(filename, format='PNG')
