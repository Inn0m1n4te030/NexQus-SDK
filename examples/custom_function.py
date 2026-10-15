import numpy as np
from nexqus import Simulator, circuit


@circuit(3)
def my_function(q, angles):
    # Python loop နှင့် function များကို ပုံမှန်အတိုင်း အသုံးပြုနိုင်ပါသည်။
    for wire,angle in enumerate(angles): q.ry(angle,wire)
    q.cx(0,1).cx(1,2)


result=Simulator().run(my_function([.2,.4,np.pi/2]),seed=123)
print(result.counts)
print('Expectation ZII:',result.expectation('ZII'))
