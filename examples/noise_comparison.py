from nexqus import Simulator, Noise, NoiseModel, algorithms

q=algorithms.ghz(3)
model=NoiseModel(Noise.depolarizing(.04),Noise.amplitude_damping(.02))
# density matrix သည် သတ်မှတ်ထားသော noise model အတွက် exact တွက်ချက်မှု ဖြစ်ပါသည်။
exact=Simulator(method='density_matrix').run(q,noise=model,shots=0)
# trajectory အရေအတွက် ပိုများလျှင် ပျမ်းမျှရလဒ် ပိုတည်ငြိမ်လာပါသည်။
sampled=Simulator(method='trajectory').run(q,noise=model,trajectories=1000,shots=0,seed=7)
print('Exact probabilities:',exact.probabilities)
print('Trajectory estimate:',sampled.probabilities)
