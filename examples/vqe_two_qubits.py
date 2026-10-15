"""Small variational energy minimization, H=-ZZ-0.5 XI-0.5 IX."""
import numpy as np
from nexqus import Circuit, Parameter, QuantumFunction, Adam

a,b,c=Parameter('a'),Parameter('b'),Parameter('c')
template=Circuit(2).ry(a,0).cx(0,1).ry(b,0).ry(c,1)
terms=[(-1,QuantumFunction(template,'ZZ')),(-.5,QuantumFunction(template,'XI')),
       (-.5,QuantumFunction(template,'IX'))]
parameters={'a':.3,'b':.2,'c':.4}; opt=Adam(lr=.08)
for epoch in range(100):
    energy=0.; gradient={name:0. for name in parameters}
    for coefficient,term in terms:
        value,derivatives=term.value_and_grad(parameters)
        energy+=coefficient*value
        for name in gradient: gradient[name]+=coefficient*derivatives[name]
    if epoch%25==0: print('epoch',epoch,'energy',energy)
    parameters=opt.step(parameters,gradient)
energy=sum(coefficient*term(parameters) for coefficient,term in terms)
X=np.array([[0,1],[1,0]]); Z=np.diag([1,-1]); I=np.eye(2)
hamiltonian=-np.kron(Z,Z)-.5*np.kron(X,I)-.5*np.kron(I,X)
print('Variational energy:',energy)
print('Exact ground energy:',np.linalg.eigvalsh(hamiltonian)[0])
