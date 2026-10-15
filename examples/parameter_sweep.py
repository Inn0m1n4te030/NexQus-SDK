import numpy as np
from nexqus import circuit, Parameter, Simulator

@circuit(2)
def ansatz(q,theta):
    # circuit ကို တစ်ကြိမ်တည်ဆောက်ပြီး angle များ ပြောင်းလဲစမ်းသပ်ပါ။
    q.ry(theta,0).cx(0,1)

theta=Parameter('theta')
template=ansatz(theta)
sim=Simulator()
results=sim.run_sweep(template,[{'theta':x} for x in np.linspace(0,np.pi,5)],
                      seed=42,shots=1000,retain_state=True)
for result in results:
    print(result.metadata['bindings'],result.counts,'ZZ:',result.expectation('ZZ'))
