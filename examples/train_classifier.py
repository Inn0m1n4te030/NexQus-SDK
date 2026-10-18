"""Toy binary classification: negative versus positive scalar inputs."""
import numpy as np
import argparse
from nexqus import Circuit, Parameter, QuantumFunction, QuantumBinaryClassifier, Adam

x,w,b=Parameter('x'),Parameter('weight'),Parameter('bias')
parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--device')
args=parser.parse_args()
function=QuantumFunction(Circuit(1).ry(w*x+b,0),'Z',device=args.device)
print('Execution:',function.simulator.backend.name,function.simulator.backend.device)
model=QuantumBinaryClassifier(function,['x'],{'weight':.7,'bias':-1.0})
X=np.r_[np.linspace(-1.4,-.15,10),np.linspace(.15,1.4,10)]
y=(X>0).astype(int)
# Label 0/1 များကို expectation target -1/+1 သို့ ပြောင်းပြီး MSE ဖြင့် train ပါသည်။
model.fit(X,y,epochs=60,optimizer=Adam(lr=.05))
test_x=np.array([-1.2,-.6,-.2,.2,.6,1.2])
predicted=model.predict(test_x)
print('Test labels:',predicted)
print('Accuracy:',float(np.mean(predicted==(test_x>0))))
print('Probability-like scores:',model.predict_proba(test_x))
