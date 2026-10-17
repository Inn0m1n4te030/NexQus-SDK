"""Reusable encoding, trainable layers, reuploading and inspection (source checkout)."""
import numpy as np
from nexqus import Circuit, Parameter, QuantumFunction, templates


def main():
    x = [Parameter('x0'), Parameter('x1')]
    shape = templates.layer_shape(2, 2, rotations='YZ')
    weights = np.array([Parameter(f'w{i}') for i in range(np.prod(shape))],dtype=object).reshape(shape)
    q = templates.data_reuploading(Circuit(2),x,weights,rotations='YZ',entangler='CZ')
    values = {'x0':.2,'x1':-.4,**{f'w{i}':.1 for i in range(weights.size)}}
    value, gradient = QuantumFunction(q,'ZI').value_and_grad(values,wrt=['w0','w1'])
    print('Expectation:',value,'weight gradients:',gradient)
    print('Logical depth:',q.depth(),'gate counts:',q.count_ops(gates_only=True))
    print(q.bind(values).draw())


if __name__ == '__main__': main()
