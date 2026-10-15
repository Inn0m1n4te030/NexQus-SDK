from nexqus import Circuit,Simulator

# Measurement ရလဒ်ကို classical bit တွင် သိမ်းပြီး conditional X ဖြင့် |0> သို့ ပြန်ထားပါ။
q=Circuit(1).h(0).measure(0,'m').x(0).c_if('m',1)
result=Simulator().run(q,shots=1000,seed=7)
print('Feed-forward quantum counts:',result.counts)
print('Measurement counts:',result.classical_counts)

q=Circuit(2).h(0).cx(0,1).reset(0)
result=Simulator().run(q,shots=1000,seed=7)
print('Reset on one half of a Bell state:',result.probabilities)
print('Reset qubit state:\n',result.reduced_density([0]))
