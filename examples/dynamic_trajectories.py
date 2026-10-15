import argparse
from nexqus import Circuit,Simulator,Noise,NoiseModel

p=argparse.ArgumentParser()
p.add_argument('--backend',choices=['numpy','torch'],default='numpy')
p.add_argument('--device',default='cpu')
args=p.parse_args()
q=Circuit(2).h(0).measure(0,'m').x(1).c_if('m',1).reset(0)
model=NoiseModel(Noise.depolarizing(.02))
r=Simulator(backend=args.backend,device=args.device,method='trajectory').run(
    q,noise=model,trajectories=2048,batch_size=128,shots=1000,seed=42)
print('Quantum probabilities:',r.probabilities)
print('Classical probabilities:',r.classical_probabilities)
print('Joint sampled counts:',r.joint_counts)
