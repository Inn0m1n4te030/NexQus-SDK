"""Trainable sentence-pair classification over externally prepared angle features."""
import json
from pathlib import Path
import numpy as np
from .circuit import Circuit, positive_integer
from .parameters import Parameter
from .generative import QuantumDistribution
from .optimizers import Adam
from .nlp import sentence_pair_circuit
from .simulator import Simulator


class SentencePairClassifier:
    """Symmetric P0 classifier; inputs are (samples, 2, qubits) angles in radians.

    Text extraction/scaling is external and must be fitted on training data only.
    Identical encodings always score one. This fidelity architecture cannot learn
    arbitrary binary relations. Probability is a model score, not calibrated confidence.
    """
    def __init__(self, n_qubits=2, *, seed=7, device='cpu', backend=None):
        self.n_qubits=positive_integer(n_qubits,'n_qubits')
        if self.n_qubits>6: raise ValueError('Use at most six qubits for this small-model API')
        self.seed=seed
        self.simulator=Simulator.for_device(device,backend=backend)
        def encoder(side):
            q=Circuit(self.n_qubits)
            for layer in range(3):
                for i in range(self.n_qubits): q.ry(Parameter(f'{side}{i}'),i)
                if layer<2:
                    for i in range(self.n_qubits):
                        getattr(q,'rz' if layer==0 else 'rx')(Parameter(f'w{layer*self.n_qubits+i}'),i)
                    for i in range(self.n_qubits-1): q.cx(i,i+1)
            return q
        self.circuit=sentence_pair_circuit(encoder('a'),encoder('b'))
        self.distribution=QuantumDistribution(self.circuit,simulator=self.simulator)
        rng=np.random.default_rng(seed)
        self.parameters_={f'w{i}':float(rng.normal(0,.3)) for i in range(2*self.n_qubits)}
        self.history_=[]
        self.fitted_=False

    def _inputs(self, pairs):
        raw=np.asarray(pairs)
        if raw.dtype.kind not in 'iuf' or raw.ndim!=3 or raw.shape[1:]!=(2,self.n_qubits) or not len(raw):
            raise ValueError('Expected nonempty (samples, 2, n_qubits) real angle features')
        values=raw.astype(float)
        if not np.isfinite(values).all(): raise ValueError('Angles must be finite')
        return values

    @staticmethod
    def _labels(labels, count):
        y=np.asarray(labels)
        if y.shape!=(count,) or y.dtype.kind not in 'biuf' or not np.isin(y,[0,1]).all():
            raise ValueError('Provide one binary label per pair')
        return y.astype(float)

    def _bindings(self,pair,parameters=None):
        return dict(zip([f'{s}{i}' for s in 'ab' for i in range(self.n_qubits)],pair.ravel())) | (self.parameters_ if parameters is None else parameters)

    def _scores(self,x,parameters):
        return np.array([float(self.distribution(self._bindings(p,parameters))[0]) for p in x])

    def fit(self,pairs,labels,*,epochs=30,learning_rate=.05,validation_data=None):
        """Fresh deterministic fit; select minimum validation MSE when supplied.

        Without validation, retain the final epoch. Never uses test examples for
        checkpoint selection. History contains at most 32 uniformly spaced epochs.
        """
        epochs=positive_integer(epochs,'epochs'); x=self._inputs(pairs); y=self._labels(labels,len(x))
        validation=None
        if validation_data is not None:
            vx=self._inputs(validation_data[0]); validation=(vx,self._labels(validation_data[1],len(vx)))
        opt=Adam(lr=learning_rate); rng=np.random.default_rng(self.seed)
        params={k:float(rng.normal(0,.3)) for k in self.parameters_}
        history=[]; best=None
        for epoch in range(epochs+1):
            loss=float(np.mean((self._scores(x,params)-y)**2))
            val=None if validation is None else float(np.mean((self._scores(validation[0],params)-validation[1])**2))
            record=dict(epoch=epoch,loss=loss,validation_loss=val,parameters=dict(params))
            history.append(record)
            if validation is None or best is None or val<best['validation_loss']: best=record
            if epoch==epochs: break
            grad=dict.fromkeys(params,0.)
            for pair,target in zip(x,y):
                values,jac=self.distribution.value_and_grad(self._bindings(pair,params),wrt=list(params))
                for k in params: grad[k]+=2*(float(values[0])-target)*float(jac[k][0])/len(x)
            params=opt.step(params,grad)
        indices=np.unique(np.linspace(0,epochs,min(32,epochs+1),dtype=int))
        self.history_=[history[i] for i in indices]
        self.parameters_=dict(best['parameters']); self.selected_epoch_=best['epoch']; self.fitted_=True
        return self

    def _require_fit(self):
        if not self.fitted_: raise ValueError('Fit or load the model first')

    def predict_proba(self,pairs):
        self._require_fit(); score=self._scores(self._inputs(pairs),self.parameters_)
        return np.column_stack([1-score,score])

    def predict(self,pairs):
        return (self.predict_proba(pairs)[:,1]>=.5).astype(int)

    def explain(self,pair,*,comparison=None,title='Sentence-pair prediction'):
        """Explain actual P0 score and local encoded-feature/weight derivatives.

        Returns the native offline explorer, with bound gates, state evolution,
        comparison and recorded training playback. Derivatives are with respect
        to angles, not words. Text provenance belongs in the calling application.
        """
        self._require_fit(); x=self._inputs([pair])[0]
        cases=None if comparison is None else [dict(name='Edited sentence pair',bindings=self._bindings(self._inputs([comparison])[0]))]
        return self.circuit.explain(bindings=self._bindings(x),observable='P0',
            title=title,parameter_roles={k:'weight' if k.startswith('w') else 'feature' for k in self.circuit.parameters},
            sensitivities=True,comparisons=cases,backend=self.simulator.backend.name,
            device=self.simulator.backend.device,
            training_history=[dict(epoch=r['epoch'],bindings=self._bindings(x,r['parameters']),loss=r['loss']) for r in self.history_])

    def save(self,path):
        """Save JSON weights/history; no executable pickle or external encoder."""
        self._require_fit()
        data=dict(schema_version=1,n_qubits=self.n_qubits,seed=self.seed,parameters=self.parameters_,
                  selected_epoch=self.selected_epoch_,history=self.history_)
        Path(path).write_text(json.dumps(data,allow_nan=False,indent=2),encoding='utf-8')

    @classmethod
    def load(cls,path,*,device='cpu',backend=None):
        data=json.loads(Path(path).read_text(encoding='utf-8'))
        if data.get('schema_version')!=1: raise ValueError('Unsupported checkpoint schema')
        model=cls(data['n_qubits'],seed=data['seed'],device=device,backend=backend)
        def weights(p):
            if not isinstance(p,dict) or set(p)!=set(model.parameters_) or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not np.isfinite(v) for v in p.values()):
                raise ValueError('Invalid checkpoint weights')
            return dict(p)
        model.parameters_=weights(data['parameters'])
        history=data['history']
        if not isinstance(history,list) or not 1<=len(history)<=32: raise ValueError('Invalid checkpoint history')
        previous=-1
        for r in history:
            epoch=r['epoch']
            if isinstance(epoch,bool) or not isinstance(epoch,int) or epoch<=previous: raise ValueError('Invalid checkpoint epoch')
            previous=epoch; weights(r['parameters'])
            for k in ('loss','validation_loss'):
                v=r[k]
                if k=='validation_loss' and v is None: continue
                if isinstance(v,bool) or not isinstance(v,(int,float)) or not np.isfinite(v) or v<0: raise ValueError('Invalid checkpoint loss')
        selected=data['selected_epoch']
        if isinstance(selected,bool) or not isinstance(selected,int) or not 0<=selected<=previous: raise ValueError('Invalid selected epoch')
        model.history_=history; model.selected_epoch_=selected; model.fitted_=True
        return model
