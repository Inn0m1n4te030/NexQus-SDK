"""Analytic parameter-shift gradients for supported symbolic circuit angles."""
import numpy as np
from .circuit import Circuit, Operation
from .parameters import Parameter, bindings_dict
from .simulator import Simulator


def _names(wrt,all_names):
    if wrt is None: return tuple(all_names)
    if isinstance(wrt,(str,Parameter)): wrt=[wrt]
    names=tuple(p.args[0] if isinstance(p,Parameter) else p for p in wrt)
    if any(not isinstance(n,str) for n in names) or len(set(names))!=len(names):
        raise ValueError('wrt must contain distinct parameter names')
    if set(names)-set(all_names): raise ValueError('Unknown differentiation parameter')
    return names


class QuantumFunction:
    """An expectation-valued circuit with gate-local analytic derivatives.

    Supports symbolic RX/RY/RZ/P/CP angles and their arithmetic chain rule.
    Repeated/shared parameters are differentiated per gate, then summed.
    Uses deterministic statevector or exact density evolution, shots=0.
    No PyTorch autograd integration or sampled trajectory gradients are implied.
    """
    def __init__(self,template,observable,simulator=None,noise=None):
        if not isinstance(template,Circuit): raise TypeError('Expected a Circuit template')
        if template.dynamic: raise ValueError('Dynamic circuits are not supported by QuantumFunction gradients in v0.5')
        if not isinstance(observable,str) or len(observable)!=template.n_qubits or set(observable)-set('IXYZ'):
            raise ValueError('Observable must be a Pauli string in q0-first order')
        self.template=template
        self.observable=observable
        self.simulator=simulator if simulator is not None else Simulator()
        if not isinstance(self.simulator,Simulator): raise TypeError('Expected Simulator')
        if self.simulator.method=='trajectory':
            raise ValueError('Analytic gradients require statevector, density_matrix, or auto; not trajectories')
        self.noise=noise

    def __call__(self,bindings):
        return self._evaluate(self.template.bind(bindings))

    def _evaluate(self,circuit):
        return self.simulator.run(circuit,noise=self.noise,shots=0,retain_state=True).expectation(self.observable)

    def gradient(self,bindings,wrt=None):
        return self._compute(bindings,wrt,include_value=False)[1]

    def value_and_grad(self,bindings,wrt=None):
        return self._compute(bindings,wrt,include_value=True)

    def _compute(self,bindings,wrt,include_value):
        values=bindings_dict(bindings)
        bound=self.template.bind(values) # Validates missing and unknown bindings.
        names=_names(wrt,self.template.parameters)
        gradients={name:0.0 for name in names}
        value=self._evaluate(bound) if include_value else None
        for index,op in enumerate(self.template.operations):
            if op.kind!='parameterized': continue
            gate,expression=op.data
            if gate not in ('rx','ry','rz','phase','cp'):
                raise ValueError(f'No gradient rule for {gate}')
            angle,chain=expression.value_and_derivatives(values)
            active=[name for name in names if chain.get(name,0)!=0]
            if not active: continue
            shifted=[]
            for shift in (np.pi/2,-np.pi/2):
                replacement=Circuit(self.template.n_qubits)
                getattr(replacement,gate)(angle+shift,*op.wires)
                concrete=replacement.operations[0]
                shifted_circuit=Circuit(self.template.n_qubits)
                shifted_circuit.operations=list(bound.operations)
                # Keep original names, so gate-specific noise rules stay fixed.
                shifted_circuit.operations[index]=Operation(op.name,concrete.wires,concrete.data,concrete.kind)
                shifted.append(self._evaluate(shifted_circuit))
            local_derivative=(shifted[0]-shifted[1])/2
            for name in active: gradients[name]+=local_derivative*chain[name]
        return value,gradients
