"""Small explicit optimizers over dictionaries of real scalar parameters."""
import numpy as np
from .parameters import bindings_dict,real_scalar


def _validate(parameters,gradients):
    parameters=bindings_dict(parameters); gradients=bindings_dict(gradients)
    if not parameters or set(parameters)!=set(gradients):
        raise ValueError('Parameters and gradients must have matching nonempty keys')
    return parameters,gradients


class SGD:
    def __init__(self,lr=.05):
        self.lr=real_scalar(lr)
        if self.lr<=0: raise ValueError('lr must be positive')

    def step(self,parameters,gradients):
        parameters,gradients=_validate(parameters,gradients)
        return {k:real_scalar(v-self.lr*gradients[k]) for k,v in parameters.items()}


class Adam:
    def __init__(self,lr=.05,beta1=.9,beta2=.999,epsilon=1e-8):
        self.lr=real_scalar(lr); self.beta1=real_scalar(beta1); self.beta2=real_scalar(beta2)
        self.epsilon=real_scalar(epsilon)
        if self.lr<=0 or self.epsilon<=0 or not 0<=self.beta1<1 or not 0<=self.beta2<1:
            raise ValueError('Invalid Adam hyperparameters')
        self.reset()

    def reset(self):
        self.t=0; self.m={}; self.v={}

    def step(self,parameters,gradients):
        parameters,gradients=_validate(parameters,gradients)
        if self.t and set(parameters)!=set(self.m):
            raise ValueError('Parameter names changed; reset the optimizer first')
        step=self.t+1
        result={}; moments={}; variances={}
        for k,value in parameters.items():
            g=gradients[k]
            moments[k]=real_scalar(self.beta1*self.m.get(k,0)+(1-self.beta1)*g)
            variances[k]=real_scalar(self.beta2*self.v.get(k,0)+(1-self.beta2)*g*g)
            m=real_scalar(moments[k]/(1-self.beta1**step))
            v=real_scalar(variances[k]/(1-self.beta2**step))
            result[k]=real_scalar(value-self.lr*m/(np.sqrt(v)+self.epsilon))
        # Commit only after every value is finite; a failed step is retryable.
        self.t=step; self.m=moments; self.v=variances
        return result
