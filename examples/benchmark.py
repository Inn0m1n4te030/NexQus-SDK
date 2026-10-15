"""CLI timing and device metadata. No universal CPU/GPU speedup is assumed."""
import argparse
import json
import statistics
from nexqus import Simulator, Noise, NoiseModel, algorithms

p=argparse.ArgumentParser()
p.add_argument('--qubits',type=int,default=10)
p.add_argument('--backend',choices=['numpy','torch'],default='numpy')
p.add_argument('--device',default='cpu')
p.add_argument('--method',choices=['auto','statevector','density_matrix','trajectory'],default='auto')
p.add_argument('--noise',type=float,default=0)
p.add_argument('--precision',choices=['single','double'],default='double')
p.add_argument('--repeat',type=int,default=3)
p.add_argument('--trajectories',type=int,default=256)
args=p.parse_args()
if args.repeat<1: p.error('--repeat must be positive')
sim=Simulator(backend=args.backend,device=args.device,method=args.method,precision=args.precision)
model=NoiseModel(Noise.depolarizing(args.noise)) if args.noise else None
q=algorithms.ghz(args.qubits)
kw=dict(shots=0,noise=model,seed=17,trajectories=args.trajectories,retain_state=False)
sim.run(q,**kw) # Warmup; excludes package import and CUDA context setup.
results=[sim.run(q,**kw) for _ in range(args.repeat)]
metadata=results[-1].metadata
metadata['median_seconds']=statistics.median(r.metadata['simulation_seconds'] for r in results)
metadata['repeats']=args.repeat
print(json.dumps(metadata,indent=2))
