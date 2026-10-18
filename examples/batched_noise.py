import argparse
from nexqus import Simulator, Noise, NoiseModel, algorithms

p=argparse.ArgumentParser()
p.add_argument('--backend',choices=['numpy','torch'])
p.add_argument('--device')
p.add_argument('--batch-size',default=32,type=int)
args=p.parse_args()
sim=Simulator.for_device(args.device,backend=args.backend,method='trajectory')
q=algorithms.ghz(6)
noise=NoiseModel(Noise.depolarizing(.03),Noise.amplitude_damping(.02))
# batch ကြီးလျှင် memory ပိုလိုအပ်သော်လည်း parallel တွက်ချက်နိုင်ပါသည်။
r=sim.compile(q,noise).run(trajectories=1000,batch_size=args.batch_size,shots=1000,seed=42)
print(r.counts)
print(r.metadata)
