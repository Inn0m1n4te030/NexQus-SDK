"""Run after installation: python examples/quickstart.py"""
from nexqus import Simulator, Noise, NoiseModel, circuit, algorithms


@circuit(2)
def entangle(q):
    # ပထမ qubit ကို superposition အခြေအနေသို့ ပြောင်းပါ။
    q.h(0)
    # qubit နှစ်ခုကို entangle လုပ်ပါ။
    q.cx(0,1)


sim = Simulator()
ideal = sim.run(entangle(),shots=1000,seed=42)
print('Noiseless:',ideal.counts)
noisy = sim.run(entangle(),noise=NoiseModel(Noise.depolarizing(.03)),seed=42)
print('Noisy:',noisy.counts)
print('Metadata:',noisy.metadata)
search = sim.run(algorithms.grover(3,'101'),seed=42)
print('Grover:',search.most_likely)
