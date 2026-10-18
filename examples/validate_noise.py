"""Trajectory-vs-exact check, separate from paired CPU/GPU numerical parity."""
import argparse
import json
from nexqus import Circuit, Simulator, Noise, NoiseModel
from nexqus.validation import trajectory_confidence_bound

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--backend',choices=['numpy','torch'])
p.add_argument('--device')
p.add_argument('--trajectories',type=int,default=10000)
p.add_argument('--batch-size',type=int,default=128)
p.add_argument('--random-engine',choices=['portable','native'],default='native')
p.add_argument('--seed',type=int,default=42)
args=p.parse_args()
q=Circuit(3).h(0).cx(0,1).ry(.7,2).cx(2,1).rx(.4,0)
model=NoiseModel(Noise.depolarizing(.07),Noise.amplitude_damping(.13),Noise.phase_damping(.05))
exact=Simulator(backend='numpy',device='cpu').run(q,noise=model,shots=0)
sampled=Simulator.for_device(args.device,backend=args.backend,method='trajectory').run(q,
    noise=model,trajectories=args.trajectories,batch_size=args.batch_size,
    random_engine=args.random_engine,seed=args.seed,shots=0)
error=float(abs(exact.probabilities-sampled.probabilities).max())
bound=trajectory_confidence_bound(args.trajectories,len(exact.probabilities),alpha=.01)
if error>bound: raise AssertionError(f'Exact-channel discrepancy {error} exceeds statistical bound {bound}')
print(json.dumps(dict(check='passed',kind='statistical sanity check',max_probability_error=error,
                     max_error_bound=bound,alpha=.01,metadata=sampled.metadata),indent=2))
