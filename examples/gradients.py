import numpy as np
from nexqus import Circuit, Parameter, QuantumFunction

theta=Parameter('theta')
# တူညီသော parameter ကို gate များစွာတွင် ပြန်သုံးထားလည်း gradient တွက်နိုင်ပါသည်။
q=Circuit(1).ry(theta,0).ry(2*theta,0)
function=QuantumFunction(q,'Z')
value,gradient=function.value_and_grad({'theta':.2})
print('Expectation:',value,'expected:',np.cos(.6))
print('Gradient:',gradient,'expected:',-3*np.sin(.6))
