"""Requires a CUDA-capable PyTorch install and NVIDIA GPU."""
from nexqus import Simulator, Noise, NoiseModel, algorithms

# CUDA မရရှိပါက error ပြပါမည်။ CPU သို့ အလိုအလျောက် မပြောင်းပါ။
sim=Simulator(backend='torch',device='cuda:0',precision='single')
r=sim.run(algorithms.ghz(12),shots=1000,seed=1)
print(r.counts,r.metadata)
# Exact density matrices grow rapidly, so this example uses only five qubits.
r=sim.run(algorithms.ghz(5),noise=NoiseModel(Noise.depolarizing(.01)),seed=1)
print(r.metadata)
