"""Full-batch MSE regression over an expectation-valued QuantumFunction."""
import numpy as np
from .gradients import QuantumFunction,_names
from .optimizers import Adam
from .parameters import bindings_dict,real_scalar
from .circuit import positive_integer


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
        X=np.asarray(X,dtype=float)
        if X.ndim==1 and len(self.input_names)==1: X=X[:,None]
        if X.ndim!=2 or X.shape[1]!=len(self.input_names) or not len(X) or not np.isfinite(X).all():
            raise ValueError('X must be finite with shape (samples, input parameters)')
        return X

    def _bindings(self,row):
        return dict(self.parameters,**{name:float(v) for name,v in zip(self.input_names,row)})

    def predict(self,X):
        return np.array([self.function(self._bindings(row)) for row in self._inputs(X)])

    def fit(self,X,y,epochs=60,optimizer=None,callback=None):
        X=self._inputs(X); y=np.asarray(y,dtype=float)
        epochs=positive_integer(epochs,'epochs')
        if y.shape!=(len(X),) or not np.isfinite(y).all(): raise ValueError('y must be finite with one target per sample')
        if np.any(abs(y)>1+1e-12): raise ValueError('Scale targets to [-1,1] for a Pauli-expectation regressor')
        opt=optimizer if optimizer is not None else Adam()
        if not callable(getattr(opt,'step',None)): raise TypeError('Optimizer must provide step()')
        self.history=[]
        for epoch in range(epochs):
            gradients={name:0.0 for name in self.parameters}
            loss=0.0
            for row,target in zip(X,y):
                value,derivatives=self.function.value_and_grad(self._bindings(row),wrt=list(self.parameters))
                error=value-target
                loss+=error*error/len(X)
                for name in gradients: gradients[name]+=2*error*derivatives[name]/len(X)
            # history records the loss before this step, at the listed parameters.
            record=dict(epoch=epoch,loss=real_scalar(loss),parameters=dict(self.parameters))
            self.history.append(record)
            if callback is not None: callback(record.copy())
            self.parameters=opt.step(self.parameters,gradients)
        return self


class QuantumBinaryClassifier(QuantumRegressor):
    """Binary 0/1 labels, trained with MSE on targets mapped to -1/+1.

    predict_proba maps the circuit expectation to (1-z)/2, (1+z)/2. These
    scores are not guaranteed calibrated probabilities. Binary cross entropy
    and multiclass training are not implemented.
    """
    def fit(self,X,y,epochs=60,optimizer=None,callback=None):
        labels=np.asarray(y)
        if labels.ndim!=1 or not np.isin(labels,[0,1]).all():
            raise ValueError('Binary classifier labels must be 0 or 1')
        return super().fit(X,2*labels.astype(float)-1,epochs,optimizer,callback)

    def predict_expectation(self,X): return super().predict(X)

    def predict_proba(self,X):
        positive=np.clip((self.predict_expectation(X)+1)/2,0,1)
        return np.column_stack([1-positive,positive])

    def predict(self,X): return (self.predict_proba(X)[:,1]>=.5).astype(int)
