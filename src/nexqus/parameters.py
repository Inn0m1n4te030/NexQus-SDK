"""Named real scalar parameters and small arithmetic expression trees."""
from dataclasses import dataclass
import numbers
import numpy as np


@dataclass(frozen=True)
class Expression:
    operation: str
    args: tuple

    @property
    def names(self):
        if self.operation == 'parameter': return {self.args[0]}
        return set().union(*(a.names for a in self.args if isinstance(a,Expression)))

    def evaluate(self, values):
        if self.operation == 'parameter': return values[self.args[0]]
        args=[a.evaluate(values) if isinstance(a,Expression) else a for a in self.args]
        if self.operation == 'add': value=args[0]+args[1]
        elif self.operation == 'mul': value=args[0]*args[1]
        elif self.operation == 'div': value=args[0]/args[1]
        else: raise ValueError('Unknown parameter expression')
        return real_scalar(value)

    def value_and_derivatives(self,values):
        """Exact scalar chain rule for the supported arithmetic expressions."""
        if self.operation=='parameter':
            return real_scalar(values[self.args[0]]),{self.args[0]:1.0}
        pairs=[a.value_and_derivatives(values) if isinstance(a,Expression) else (a,{}) for a in self.args]
        (a,da),(b,db)=pairs
        keys=set(da)|set(db)
        if self.operation=='add':
            value=a+b; derivatives={k:da.get(k,0)+db.get(k,0) for k in keys}
        elif self.operation=='mul':
            value=a*b; derivatives={k:da.get(k,0)*b+a*db.get(k,0) for k in keys}
        elif self.operation=='div':
            value=a/b; derivatives={k:(da.get(k,0)*b-a*db.get(k,0))/(b*b) for k in keys}
        else: raise ValueError('Unknown parameter expression')
        return real_scalar(value),{k:real_scalar(v) for k,v in derivatives.items()}

    def __add__(self, other): return Expression('add',(self,operand(other)))
    def __radd__(self, other): return self+other
    def __neg__(self): return self * -1
    def __sub__(self, other): return self + (-operand(other))
    def __rsub__(self, other): return -self + other
    def __mul__(self, other): return Expression('mul',(self,operand(other)))
    def __rmul__(self, other): return self*other
    def __truediv__(self, other): return Expression('div',(self,operand(other)))
    def __rtruediv__(self, other): return Expression('div',(operand(other),self))


class Parameter(Expression):
    def __init__(self,name):
        if not isinstance(name,str) or not name.strip():
            raise ValueError('Parameter name must be a nonempty string')
        super().__init__('parameter',(name,))


def real_scalar(value):
    if isinstance(value,bool) or not isinstance(value,numbers.Real) or not np.isfinite(value):
        raise ValueError('Angles and parameter bindings must be finite real scalars')
    return float(value)


def operand(value):
    return value if isinstance(value,Expression) else real_scalar(value)


def bindings_dict(mapping):
    result={}
    for key,value in mapping.items():
        if isinstance(key,Parameter): key=key.args[0]
        if not isinstance(key,str): raise ValueError('Binding keys must be names or Parameters')
        if key in result: raise ValueError(f'Duplicate binding for {key}')
        result[key]=real_scalar(value)
    return result
