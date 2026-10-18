"""Full-batch MSE regression over an expectation-valued QuantumFunction."""
import numpy as np
from .gradients import QuantumFunction,_names
from .optimizers import Adam
from .parameters import bindings_dict,real_scalar
from .circuit import positive_integer


def _real_array(values, name):
    if np.iscomplexobj(values):
        raise ValueError(f'{name} must be real; complex values are not supported')
    return np.asarray(values,dtype=float)


class QuantumRegressor:
    """Train circuit parameters with analytic gradients; one scalar target/sample.

    input_names identify circuit parameters supplied from X, in column order.
    All other template parameters must have initial trainable values. A Pauli
    expectation is bounded to [-1,1]; scale real data targets appropriately.
    This is explicit optimization, not a Torch neural-network layer.
    """
    def __init__(self,function,input_names,initial_parameters):
        if not isinstance(function,QuantumFunction): raise TypeError('Expected QuantumFunction')
        self.function=function
        self.input_names=_names(input_names,function.template.parameters)
        if not self.input_names: raise ValueError('At least one input parameter is required')
        self.parameters=bindings_dict(initial_parameters)
        if set(self.parameters)&set(self.input_names): raise ValueError('Inputs cannot also be trainable parameters')
        if set(self.parameters)|set(self.input_names)!=set(function.template.parameters) or not self.parameters:
            raise ValueError('Supply an initial value for every non-input template parameter')
        self.history=[]

    def _inputs(self,X):
        X=_real_array(X,'X')
        if X.ndim==1 and len(self.input_names)==1: X=X[:,None]
        if X.ndim!=2 or X.shape[1]!=len(self.input_names) or not len(X) or not np.isfinite(X).all():
            raise ValueError('X must be finite with shape (samples, input parameters)')
        return X

    def _bindings(self,row):
        return dict(self.parameters,**{name:float(v) for name,v in zip(self.input_names,row)})

    def predict(self,X):
        X=self._inputs(X)
        if self._tensor_enabled():
            circuit,inputs=self._tensor_inputs(X)
            import torch
            with torch.no_grad():
                return circuit.expectation(dict(inputs,**self.parameters),self.function.observable).cpu().numpy()
        return np.array([self.function(self._bindings(row)) for row in X])

    def _tensor_enabled(self):
        return bool(self.function.simulator.backend.torch and self.function.noise is None
                    and not any(op.kind=='noise' for op in self.function.template.operations))

    def _tensor_inputs(self,X):
        from .torch_circuit import TorchCircuit
        sim=self.function.simulator
        circuit=TorchCircuit(self.function.template,device=sim.backend.device,precision=sim.precision,
                             max_memory_mb=sim.max_memory_mb,max_qubits=self.function.template.n_qubits)
        t=sim.backend.torch
        inputs=t.tensor(X,device=sim.backend.device,dtype=circuit._anchor.dtype)
        return circuit,{name:inputs[:,i] for i,name in enumerate(self.input_names)}

    def fit(self,X,y,epochs=60,optimizer=None,callback=None):
        X=self._inputs(X); y=_real_array(y,'y')
        epochs=positive_integer(epochs,'epochs')
        if y.shape!=(len(X),) or not np.isfinite(y).all(): raise ValueError('y must be finite with one target per sample')
        if np.any(abs(y)>1+1e-12): raise ValueError('Scale targets to [-1,1] for a Pauli-expectation regressor')
        opt=optimizer if optimizer is not None else Adam()
        if not callable(getattr(opt,'step',None)): raise TypeError('Optimizer must provide step()')
        self.history=[]
        tensor_circuit,inputs=self._tensor_inputs(X) if self._tensor_enabled() else (None,None)
        tensor_target=None
        if tensor_circuit is not None:
            tensor_target=self.function.simulator.backend.torch.as_tensor(y,device=tensor_circuit._anchor.device,dtype=tensor_circuit._anchor.dtype)
        for epoch in range(epochs):
            gradients={name:0.0 for name in self.parameters}
            loss=0.0
            if tensor_circuit is not None:
                t=self.function.simulator.backend.torch
                params={k:t.tensor(v,device=tensor_circuit._anchor.device,dtype=tensor_circuit._anchor.dtype,requires_grad=True)
                        for k,v in self.parameters.items()}
                predicted=tensor_circuit.expectation(dict(inputs,**params),self.function.observable)
                tensor_loss=((predicted-tensor_target)**2).mean()
                derivatives=t.autograd.grad(tensor_loss,tuple(params.values()))
                gradients={k:float(v) for k,v in zip(params,derivatives)}
                loss=float(tensor_loss.detach())
            else:
                for row,target in zip(X,y):
                    value,derivatives=self.function.value_and_grad(self._bindings(row),wrt=list(self.parameters))
                    error=value-target
                    loss+=error*error/len(X)
                    for name in gradients: gradients[name]+=2*error*derivatives[name]/len(X)
            # history records the loss before this step, at the listed parameters.
            record=dict(epoch=epoch,loss=real_scalar(loss),parameters=dict(self.parameters))
            self.history.append(record)
            if callback is not None: callback(dict(record,parameters=dict(record['parameters'])))
            self.parameters=opt.step(self.parameters,gradients)
        return self


class QuantumBinaryClassifier(QuantumRegressor):
    """Binary 0/1 labels, trained with MSE on targets mapped to -1/+1.

    predict_proba maps the circuit expectation to (1-z)/2, (1+z)/2. These
    scores are not guaranteed calibrated probabilities. Binary cross entropy
    and multiclass training are not implemented.
    """
    def fit(self,X,y,epochs=60,optimizer=None,callback=None):
        labels=_real_array(y,'y')
        if labels.ndim!=1 or not np.isin(labels,[0,1]).all():
            raise ValueError('Binary classifier labels must be 0 or 1')
        return super().fit(X,2*labels.astype(float)-1,epochs,optimizer,callback)

    def predict_expectation(self,X): return super().predict(X)

    def predict_proba(self,X):
        positive=np.clip((self.predict_expectation(X)+1)/2,0,1)
        return np.column_stack([1-positive,positive])

    def predict(self,X): return (self.predict_proba(X)[:,1]>=.5).astype(int)
