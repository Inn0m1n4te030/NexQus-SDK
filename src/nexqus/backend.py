"""Numerical adapters. Circuit evolution is implemented by NexQus itself."""
import numpy as np
from math import prod, sqrt


class Backend:
    def __init__(self, backend="numpy", device="cpu", precision="double", local_gates=True):
        if backend not in ("numpy", "torch"):
            raise ValueError("backend must be 'numpy' or 'torch'")
        if precision not in ("single", "double"):
            raise ValueError("precision must be 'single' or 'double'")
        self.name = backend
        self.device = device
        self.itemsize = 8 if precision == "single" else 16
        self.torch = None
        self.local_gates=bool(local_gates)
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
            host_dtype=np.complex64 if self.itemsize==8 else np.complex128
            host=np.array(x,dtype=host_dtype,copy=True)
            return self.torch.from_numpy(host).to(device=self.device,dtype=self.dtype)
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

    def apply(self, tensor, data, targets, diagonal=False, permutation=False):
        """Move target tensor axes to front; apply a local operator, then restore.

        No full 2**n square gate matrix is constructed.
        """
        if self.local_gates and diagonal:
            return tensor*self._broadcast_diagonal(data,targets,tensor.ndim)
        if self.local_gates and not permutation and len(targets)==1 and self.torch is None:
            return self._local_one(tensor,data,targets[0])
        axes = tuple(targets) + tuple(i for i in range(tensor.ndim) if i not in targets)
        front = self.permute(tensor, axes).reshape(2 ** len(targets), -1)
        if permutation:
            out = self.torch.index_select(front, 0, data) if self.torch else front[data]
        else:
            out = data[:, None] * front if diagonal else data @ front
        restore = tuple(int(i) for i in np.argsort(axes))
        return self.permute(out.reshape((2,) * tensor.ndim), restore)

    def apply_batch(self, tensor, data, targets, diagonal=False, permutation=False):
        """Apply a shared gate to B statevectors with shape (B, 2, ..., 2)."""
        target_axes=tuple(q+1 for q in targets)
        if self.local_gates and diagonal:
            return tensor*self._broadcast_diagonal(data,target_axes,tensor.ndim)
        if self.local_gates and not permutation and len(targets)==1 and self.torch is None:
            return self._local_one(tensor,data,target_axes[0])
        axes=(0,)+target_axes+tuple(i for i in range(1,tensor.ndim) if i not in target_axes)
        shape=tensor.shape
        front=self.permute(tensor,axes).reshape(shape[0],2**len(targets),-1)
        if permutation:
            out = self.torch.index_select(front, 1, data) if self.torch else front[:, data, :]
        else:
            out=data[None,:,None]*front if diagonal else data @ front
        restore=tuple(int(i) for i in np.argsort(axes))
        return self.permute(out.reshape(tuple(shape[i] for i in axes)),restore)

    def _broadcast_diagonal(self,data,targets,ndim):
        order=tuple(int(i) for i in np.argsort(targets))
        ordered=self.permute(data.reshape((2,)*len(targets)),order)
        shape=tuple(2 if i in targets else 1 for i in range(ndim))
        return ordered.reshape(shape)

    def _local_one(self,tensor,data,axis):
        """NumPy pairwise updates: avoid BLAS threading and target-axis transposes."""
        # Includes numeric RZ gates and custom diagonal 2x2 operators. Check
        # matrix values, never names. Avoid multiplying the zero off-diagonals.
        if data[0,1] == 0 and data[1,0] == 0:
            return tensor*self._broadcast_diagonal(data.diagonal(),(axis,),tensor.ndim)
        left=prod(tensor.shape[:axis])
        right=prod(tensor.shape[axis+1:])
        x=tensor.reshape(left,2,right)
        out=np.empty(x.shape,dtype=self.dtype)
        if not np.any(data.imag): data=data.real
        # Reuse one half-state scratch buffer instead of allocating expression
        # temporaries for every multiply/add. Input state is never overwritten.
        scratch=np.empty((left,right),dtype=self.dtype)
        np.multiply(data[0,0],x[:,0,:],out=out[:,0,:])
        np.multiply(data[0,1],x[:,1,:],out=scratch)
        np.add(out[:,0,:],scratch,out=out[:,0,:])
        np.multiply(data[1,0],x[:,0,:],out=out[:,1,:])
        np.multiply(data[1,1],x[:,1,:],out=scratch)
        np.add(out[:,1,:],scratch,out=out[:,1,:])
        return out.reshape(tensor.shape)

    def density_channel(self,tensor,wire,n,pauli=None,damping=None):
        """NumPy exact one-wire channels using density blocks; None falls back.

        Coefficients come from validated Kraus values, including renamed/custom
        Pauli mixtures. Arbitrary channels and Torch use generic Kraus evolution.
        """
        if self.torch or (pauli is None and damping is None): return None
        blocks=[]
        for a,b in ((0,0),(0,1),(1,0),(1,1)):
            index=[slice(None)]*tensor.ndim
            index[wire],index[n+wire]=a,b
            blocks.append(tuple(index))
        i00,i01,i10,i11=blocks
        a,b,c,d=(tensor[index] for index in blocks)
        out=np.empty(tensor.shape,dtype=self.dtype)
        if pauli is not None:
            identity,x,y,z=pauli
            keep,flip=identity+z,x+y
            same,cross=identity-z,x-y
            out[i00]=keep*a+flip*d
            out[i11]=keep*d+flip*a
            out[i01]=same*b+cross*c
            out[i10]=same*c+cross*b
        else:
            kind,gamma=damping
            out[i00]=a+gamma*d if kind=='amplitude' else a
            out[i11]=(1-gamma)*d if kind=='amplitude' else d
            out[i01]=sqrt(1-gamma)*b
            out[i10]=sqrt(1-gamma)*c
        return out

    def index_array(self, values):
        indices = np.array(values, dtype=np.int64, copy=True)
        if self.torch:
            return self.torch.from_numpy(indices).to(device=self.device)
        return indices

    def real_array(self,x):
        if self.torch:
            return self.torch.from_numpy(np.array(x,dtype=np.float64,copy=True)).to(device=self.device)
        return np.array(x,dtype=np.float64,copy=True)

    def real_zeros(self,shape):
        if self.torch: return self.torch.zeros(shape,dtype=self.torch.float64,device=self.device)
        return np.zeros(shape,dtype=np.float64)

    def expand_state(self,state,count):
        if self.torch: return state[None,:].expand(count,-1)
        return np.broadcast_to(state,(count,state.shape[0]))

    def apply_cx(self,tensor,control,target):
        """Swap target amplitude slices only within the control=1 subspace."""
        out=tensor.clone() if self.torch else tensor.copy()
        zero=[slice(None)]*tensor.ndim; one=list(zero)
        zero[control]=one[control]=1
        zero[target]=0; one[target]=1
        out[tuple(zero)]=tensor[tuple(one)]
        out[tuple(one)]=tensor[tuple(zero)]
        return out

    def stack(self,values,axis=0):
        return self.torch.stack(values,dim=axis) if self.torch else np.stack(values,axis=axis)

    def where(self,condition,a,b):
        return self.torch.where(condition,a,b) if self.torch else np.where(condition,a,b)

    def clamp_min(self,x,minimum):
        return x.clamp_min(minimum) if self.torch else np.maximum(x,minimum)

    def cumulative(self,x,axis):
        return self.torch.cumsum(x,dim=axis) if self.torch else np.cumsum(x,axis=axis)
