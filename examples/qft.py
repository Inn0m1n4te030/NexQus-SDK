import numpy as np
from nexqus import Simulator, algorithms

# classical vector ကို amplitude encoding ဖြင့် normalized state အဖြစ် ပြင်ဆင်ပါ။
vector=np.array([1,2,3,4,0,0,0,0],dtype=complex)
vector/=np.linalg.norm(vector)
out=Simulator().run(algorithms.qft(3),initial_state=vector,shots=0)
print('QFT complex amplitudes:',out.statevector)
print('Measurement probabilities:',out.probabilities)
# Full complex amplitudes are simulator access, not directly measurable on hardware.
