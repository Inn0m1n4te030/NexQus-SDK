"""Learn sin(x) using an expectation-valued one-qubit circuit."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from nexqus import Circuit, Parameter, Simulator, QuantumFunction, QuantumRegressor, Adam

p=argparse.ArgumentParser()
p.add_argument('--backend',choices=['numpy','torch'])
p.add_argument('--device',default=None)
p.add_argument('--epochs',type=int,default=80)
p.add_argument('--output-dir',type=Path)
args=p.parse_args()
x,w,b=Parameter('x'),Parameter('weight'),Parameter('bias')
template=Circuit(1).ry(w*x+b,0)
simulator=Simulator.for_device(args.device,backend=args.backend)
function=QuantumFunction(template,'Z',simulator)
model=QuantumRegressor(function,['x'],{'weight':.7,'bias':-.8})
X=np.linspace(-np.pi,np.pi,20); y=np.sin(X)
initial_mse=float(np.mean((model.predict(X)-y)**2))
# parameter-shift gradient ဖြင့် circuit ၏ weight နှင့် bias ကို သင်ယူပါ။
model.fit(X,y,epochs=args.epochs,optimizer=Adam(lr=.05))
test_x=np.linspace(-np.pi,np.pi,101); predicted=model.predict(test_x)
test_mse=float(np.mean((predicted-np.sin(test_x))**2))
report=dict(initial_train_mse=initial_mse,final_train_mse=float(np.mean((model.predict(X)-y)**2)),
            test_mse=test_mse,parameters=model.parameters,epochs=args.epochs,
            backend=simulator.backend.name,device=simulator.backend.device,
            note='Synthetic learning demonstration')
print(json.dumps(report,indent=2))
if args.output_dir:
    args.output_dir.mkdir(parents=True,exist_ok=True)
    (args.output_dir/'training.json').write_text(json.dumps(dict(report,history=model.history),indent=2)+'\n')
    with (args.output_dir/'predictions.csv').open('w',newline='') as handle:
        writer=csv.writer(handle); writer.writerow(['x','target','prediction'])
        writer.writerows(zip(test_x,np.sin(test_x),predicted))
