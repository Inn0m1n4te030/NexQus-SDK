"""Optional differentiable, batched NexQus pure-state execution on Torch CPU/CUDA.

Circuit structure is compiled on the host. Bound angles, gates, state evolution,
probabilities and autograd remain on the selected device. No NumPy execution
or parameter-to-host conversion occurs in forward/backward.
"""
import copy
import os
import numpy as np
try:
    import torch
    from torch import nn
except ImportError as exc:
    raise ImportError('Install PyTorch to use nexqus.torch_circuit') from exc
from .backend import Backend
from .circuit import Circuit, positive_integer
from .parameters import Expression, Parameter


def tensor_device(device='cpu'):
    if device is None: device=os.environ.get('NEXQUS_DEVICE','cpu')
    if str(device)=='auto': device='cuda:0' if torch.cuda.is_available() else 'cpu'
    b=Backend('torch',str(device))
    return torch.empty(0,device=b.device).device


def _expression(expression, bindings):
    if not isinstance(expression,Expression): return expression
    if expression.operation=='parameter': return bindings[expression.args[0]]
    a,b=(_expression(v,bindings) for v in expression.args)
    if expression.operation=='add': return a+b
    if expression.operation=='mul': return a*b
    if expression.operation=='div': return a/b
    raise ValueError('Unsupported parameter expression')


class TorchCircuit(nn.Module):
    """Static pure-state circuit with shared scalars or per-sample angle vectors.

    Supports RX/RY/RZ/P/CP expressions, fixed unitaries, diagonals and
    permutations. Noise/measure/reset require Simulator's channel/dynamic APIs.
    .to('cuda:N') and .float()/.double() move all registered circuit tensors.
    The state dimension is bounded before allocation, including the batch.
    """
    def __init__(self,circuit,*,max_qubits=20,max_memory_mb=1024,device='cpu',precision='double'):
        super().__init__()
        if not isinstance(circuit,Circuit): raise TypeError('Expected Circuit')
        if circuit.n_qubits>positive_integer(max_qubits,'max_qubits'): raise ValueError('Circuit exceeds max_qubits')
        if circuit.dynamic or any(op.kind not in ('unitary','diagonal','permutation','parameterized') for op in circuit.operations):
            raise ValueError('TorchCircuit requires a static pure-state circuit')
        if precision not in ('single','double'): raise ValueError('precision must be single or double')
        if not np.isfinite(max_memory_mb) or max_memory_mb<=0: raise ValueError('max_memory_mb must be positive')
        self.template=copy.deepcopy(circuit); self.n_qubits=circuit.n_qubits
        self.max_memory_mb=float(max_memory_mb)
        self.register_buffer('_anchor',torch.empty(0,dtype=torch.float32 if precision=='single' else torch.float64))
        for i,op in enumerate(self.template.operations):
            if op.kind=='parameterized':
                if op.data[0] not in ('rx','ry','rz','phase','cp'): raise ValueError('Unsupported symbolic gate')
            elif op.kind=='permutation': self.register_buffer(f'g{i}',torch.tensor(np.argsort(op.data),dtype=torch.long))
            else:
                data=np.asarray(op.data,complex)
                self.register_buffer(f'g{i}',torch.tensor(np.stack([data.real,data.imag],axis=-1),dtype=self._anchor.dtype))
        self.to(tensor_device(device))

    def _apply_gate(self,state,matrix,wires,*,diagonal=False,permutation=False):
        n=self.n_qubits; batch=state.shape[0]
        axes=(0,)+tuple(q+1 for q in wires)+tuple(q+1 for q in range(n) if q not in wires)
        front=state.reshape((batch,)+(2,)*n).permute(axes).reshape(batch,2**len(wires),-1)
        if permutation: out=front.index_select(1,matrix)
        elif diagonal:
            out=front*(matrix[None,:,None] if matrix.ndim==1 else matrix[:,:,None])
        else: out=matrix@front
        inverse=tuple(axes.index(i) for i in range(len(axes)))
        return out.reshape((batch,)+(2,)*n).permute(inverse).reshape(batch,-1)

    def statevector(self,bindings):
        if self._anchor.dtype not in (torch.float32,torch.float64): raise ValueError('TorchCircuit requires float32 or float64')
        values={k.args[0] if isinstance(k,Parameter) else k:v for k,v in bindings.items()}
        if len(values)!=len(bindings) or set(values)!=set(self.template.parameters): raise ValueError('Bindings must exactly match circuit parameters')
        batch=1
        for name,v in list(values.items()):
            if not torch.is_tensor(v):
                if isinstance(v,bool) or np.iscomplexobj(v): raise ValueError('Angles must be real')
                v=torch.as_tensor(v,dtype=self._anchor.dtype,device=self._anchor.device)
            if v.device!=self._anchor.device or v.dtype!=self._anchor.dtype or v.ndim>1 or not torch.isfinite(v).all():
                raise ValueError('Bindings must be finite scalar/1D tensors matching circuit device and float dtype')
            if v.ndim:
                if len(v)<1: raise ValueError('Empty angle batch')
            values[name]=v
        sizes={len(v) for v in values.values() if v.ndim}
        if len(sizes)>1: raise ValueError('Inconsistent angle batch sizes')
        batch=next(iter(sizes),1)
        real=self._anchor.dtype
        complex_dtype=torch.complex64 if real==torch.float32 else torch.complex128
        if batch*2**self.n_qubits*(8 if real==torch.float32 else 16)*6>self.max_memory_mb*1024**2:
            raise MemoryError('Batched state exceeds estimated max_memory_mb')
        state=torch.zeros(batch,2**self.n_qubits,dtype=complex_dtype,device=self._anchor.device); state[:,0]=1
        for i,op in enumerate(self.template.operations):
            if op.kind=='parameterized':
                gate,expression=op.data; angle=_expression(expression,values)
                if not torch.isfinite(angle).all(): raise ValueError('Nonfinite angle expression')
                angle=angle.expand(batch); c=torch.cos(angle/2).to(complex_dtype); s=torch.sin(angle/2).to(complex_dtype)
                one=torch.ones_like(c); zero=torch.zeros_like(c)
                if gate=='rx': matrix=torch.stack([c,-1j*s,-1j*s,c],dim=1).reshape(batch,2,2)
                elif gate=='ry': matrix=torch.stack([c,-s,s,c],dim=1).reshape(batch,2,2)
                elif gate=='rz': matrix=torch.stack([torch.exp(-.5j*angle),torch.exp(.5j*angle)],dim=1)
                elif gate=='phase': matrix=torch.stack([one,torch.exp(1j*angle)],dim=1)
                else: matrix=torch.stack([one,one,one,torch.exp(1j*angle)],dim=1)
                state=self._apply_gate(state,matrix,op.wires,diagonal=gate in ('rz','phase','cp'))
            else:
                stored=getattr(self,f'g{i}')
                matrix=stored if op.kind=='permutation' else torch.view_as_complex(stored.contiguous())
                state=self._apply_gate(state,matrix,op.wires,diagonal=op.kind=='diagonal',permutation=op.kind=='permutation')
        return state

    def forward(self,bindings):
        state=self.statevector(bindings)
        return (state.conj()*state).real

    def expectation(self,bindings,pauli):
        from .circuit import X,Y,Z
        if not isinstance(pauli,str) or len(pauli)!=self.n_qubits or set(pauli)-set('IXYZ'): raise ValueError('Invalid Pauli string')
        state=self.statevector(bindings); work=state
        for q,p in enumerate(pauli):
            if p!='I': work=self._apply_gate(work,torch.tensor({'X':X,'Y':Y,'Z':Z}[p],dtype=state.dtype,device=state.device),(q,))
        return (state.conj()*work).sum(1).real
