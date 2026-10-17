"""Generate three offline HTML explainability reports; no cloud calls."""
import argparse
from pathlib import Path
import numpy as np
from nexqus import Circuit, Parameter, QuantumFunction, Adam


def generate(output):
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    bell=Circuit(2).h(0).cx(0,1)
    bell.explain(title='Bell state · where entanglement appears',observable='ZZ',
        blocks=[dict(name='Create superposition',start=0,stop=1,role='encoding'),
                dict(name='Entangle',start=1,stop=2,role='entangling')]
    ).to_html(filename=output/'bell.html')
    interference=Circuit(1).h(0).z(0).h(0)
    interference.explain(title='Interference · a phase change becomes a probability change'
    ).to_html(filename=output/'interference.html')

    x=Parameter('x'); weights=[Parameter(f'w{i}') for i in range(4)]
    q=Circuit(2).ry(x,0).ry(.5*x,1)
    q.ry(weights[0],0).ry(weights[1],1).cx(0,1).ry(weights[2],0).ry(weights[3],1)
    model=QuantumFunction(q,'ZZ'); params=dict(w0=.3,w1=-.2,w2=.15,w3=.4)
    optimizer=Adam(lr=.05); train_x=np.linspace(-2,2,12)
    def loss(values):
        return float(np.mean([(model(dict(values,x=float(v)))-np.cos(v))**2 for v in train_x]))
    before=loss(params)
    for _ in range(40):
        gradient=dict.fromkeys(params,0.)
        for v in train_x:
            prediction,derivatives=model.value_and_grad(dict(params,x=float(v)),wrt=list(params))
            for name in params:
                gradient[name]+=2*(prediction-np.cos(v))*derivatives[name]/len(train_x)
        params=optimizer.step(params,gradient)
    after=loss(params)
    if not after<before: raise RuntimeError('Example QNN training failed to reduce its loss')
    q.explain(title='Trained QNN · trace one input through encoding and learned layers',
        bindings=dict(params,x=.7),observable='ZZ',
        parameter_roles=dict(x='feature',**{k:'weight' for k in params}),
        blocks=[dict(name='Angle encoding',start=0,stop=2,role='encoding'),
                dict(name='Trainable rotations',start=2,stop=4,role='trainable'),
                dict(name='Entangling operation',start=4,stop=5,role='entangling'),
                dict(name='Readout rotations',start=5,stop=7,role='trainable')]
    ).to_html(filename=output/'qnn.html')
    print(f'QNN training MSE: {before:.6g} -> {after:.6g} (synthetic cos(x), 40 Adam updates)')
    print('QNN output is the ZZ expectation itself; no classifier transform is applied.')
    print('Saved local reports:',output.resolve())


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('.release-artifacts/explanations'))
    generate(parser.parse_args().output)
