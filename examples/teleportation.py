"""Teleport an arbitrary pure state from q0 to q2 with measured feed-forward."""
import argparse
import numpy as np
from nexqus import Circuit,Simulator,algorithms

p=argparse.ArgumentParser()
p.add_argument('--backend',choices=['numpy','torch'])
p.add_argument('--device')
args=p.parse_args()
theta=.83; phi=-.47
q=Circuit(3).ry(theta,0).phase(phi,0)
# Alice ၏ qubit နှစ်ခုကို measure လုပ်ပြီး ရလဒ်အလိုက် Bob ကို correction ပေးပါ။
q.compose(algorithms.teleportation())
result=Simulator.for_device(args.device,backend=args.backend).run(q,shots=1000,seed=42)
target=np.array([np.cos(theta/2),np.exp(1j*phi)*np.sin(theta/2)])
bob=result.reduced_density([2])
fidelity=float(np.vdot(target,bob@target).real)
print('Classical probabilities:',result.classical_probabilities)
print('Bob reduced density matrix:\n',bob)
print('Teleportation fidelity:',fidelity)
print('Joint counts (classical|quantum):',result.joint_counts)
if not np.isclose(fidelity,1,atol=1e-10): raise AssertionError('Teleportation failed')
