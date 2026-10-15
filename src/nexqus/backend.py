"""Numerical adapters. Circuit evolution is implemented by NexQus itself."""
import numpy as np


class Backend:
    def __init__(self, backend="numpy", device="cpu", precision="double"):
        if backend not in ("numpy", "torch"):
            raise ValueError("backend must be 'numpy' or 'torch'")
        if precision not in ("single", "double"):
            raise ValueError("precision must be 'single' or 'double'")
        self.name = backend
        self.device = device
        self.itemsize = 8 if precision == "single" else 16
        self.torch = None
        if backend == "numpy":
            if device != "cpu":
                raise ValueError("NumPy runs on CPU. For CUDA use backend='torch'.")
            self.dtype = np.complex64 if precision == "single" else np.complex128
        else:
            try:
                import torch
            except ImportError as exc:
                raise ImportError("Install PyTorch to use backend='torch'; see README GPU setup") from exc
            self.torch = torch
            if device != "cpu" and device != "cuda" and not device.startswith("cuda:"):
                raise ValueError("Supported devices: cpu, cuda, cuda:N")
            if device.startswith("cuda") and not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but unavailable; no silent CPU fallback")
            self.dtype = torch.complex64 if precision == "single" else torch.complex128
            # Fail early for an invalid CUDA index.
            torch.empty(0, device=device)

    def array(self, x):
        if self.torch:
            return self.torch.tensor(np.array(x, copy=True), dtype=self.dtype, device=self.device)
        return np.array(x, dtype=self.dtype, copy=True)

    def zeros(self, shape):
        if self.torch: return self.torch.zeros(shape, dtype=self.dtype, device=self.device)
        return np.zeros(shape, dtype=self.dtype)

    def permute(self, x, axes):
        return x.permute(*axes) if self.torch else np.transpose(x, axes)

    def numpy(self, x):
        return x.detach().cpu().numpy() if self.torch else np.asarray(x)

    def sync(self):
        if self.torch and self.device.startswith("cuda"):
            self.torch.cuda.synchronize(self.device)

    def apply(self, tensor, data, targets, diagonal=False):
        """Move target tensor axes to front; apply a local operator, then restore.

        No full 2**n square gate matrix is constructed.
        """
        axes = tuple(targets) + tuple(i for i in range(tensor.ndim) if i not in targets)
        front = self.permute(tensor, axes).reshape(2 ** len(targets), -1)
        out = data[:, None] * front if diagonal else data @ front
        restore = tuple(int(i) for i in np.argsort(axes))
        return self.permute(out.reshape((2,) * tensor.ndim), restore)
