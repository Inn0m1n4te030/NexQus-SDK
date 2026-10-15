from nexqus import Simulator, Noise, NoiseModel, algorithms

sim=Simulator()
# device tensor များကို ကြိုတင်ပြင်ဆင်ပြီး အကြိမ်များစွာ ပြန်သုံးပါ။
prepared=sim.compile(algorithms.ghz(5),noise=NoiseModel(Noise.depolarizing(.02)))
for seed in range(3):
    result=prepared.run(seed=seed,shots=1000)
    print(seed,result.counts,result.metadata['simulation_seconds'])
print('Gate cache:',sim.cache_info())
