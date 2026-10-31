"""Train/evaluate a small CPU QNLP fixture with matched classical baselines."""
import argparse
import csv
import hashlib
import html
import json
from pathlib import Path
import time
import platform
import sys
import numpy as np
from nexqus import Circuit, Parameter, QuantumDistribution, Adam, nlp
from nlp_similarity import SentenceFeatures


def load_pairs(path):
    with Path(path).open(encoding='utf-8') as handle:rows=list(csv.DictReader(handle))
    seen=set()
    for row in rows:
        if set(row)!={'split','left','right','label','category'} or row['split'] not in ('train','validation','test'):
            raise ValueError('CSV needs split,left,right,label,category and train/validation/test splits')
        if row['label'] not in ('0','1') or not all(row[k].strip() for k in ('left','right','category')):
            raise ValueError('Each row needs two sentences, a binary label and category')
        key=tuple(sorted(' '.join(row[k].casefold().split()) for k in ('left','right')))
        if key in seen:raise ValueError('Duplicate or reversed pair across dataset')
        seen.add(key);row['label']=int(row['label'])
    for split in ('train','validation','test'):
        selected=[r for r in rows if r['split']==split]
        if {r['label'] for r in selected}!={0,1}:raise ValueError('Every split needs both labels')
    return rows


def pair_template(n=2):
    def encoder(side):
        q=Circuit(n)
        for i in range(n):q.ry(Parameter(f'{side}{i}'),i)
        # Trainable noncommuting rotations are BETWEEN feature uploads.
        for i in range(n):q.rz(Parameter(f'w{i}'),i)
        q.cx(0,1)
        for i in range(n):q.ry(Parameter(f'{side}{i}'),i)
        for i in range(n):q.rx(Parameter(f'w{n+i}'),i)
        q.cx(0,1)
        for i in range(n):q.ry(Parameter(f'{side}{i}'),i)
        return q
    left,right=encoder('a'),encoder('b')
    return nlp.sentence_pair_circuit(left,right)


def metrics(y,p):
    y=np.asarray(y);p=np.asarray(p);pred=p>=.5
    tp=int(np.sum(pred&(y==1)));fp=int(np.sum(pred&(y==0)));fn=int(np.sum(~pred&(y==1)))
    return dict(mse=float(np.mean((p-y)**2)),accuracy=float(np.mean(pred==y)),
                f1=float(2*tp/(2*tp+fp+fn)) if 2*tp+fp+fn else 0.,samples=len(y))


def classical_features(a,b):return np.r_[abs(a-b),a*b]


def classical_value_grad(x,params,kind):
    values=np.array(list(params.values()))
    if kind=='linear':
        logit=x@values[:-1]+values[-1];p=1/(1+np.exp(-np.clip(logit,-40,40)))
        gradient=p*(1-p)*np.r_[x,1]
    else:
        w=values[:32].reshape(4,8);bias=values[32:40];v=values[40:48]
        hidden=np.tanh(x@w+bias);logit=hidden@v+values[48]
        p=1/(1+np.exp(-np.clip(logit,-40,40)));delta=p*(1-p);d=delta*v*(1-hidden**2)
        gradient=np.r_[np.outer(x,d).ravel(),d,delta*hidden,delta]
    return float(p),dict(zip(params,gradient))


def fit_model(kind,splits,seed,epochs,lr):
    if kind=='majority':
        p=float(np.mean([r['label'] for r in splits['train']]))
        predictions={s:[p]*len(rows) for s,rows in splits.items()}
        return dict(kind=kind,seed=seed,parameters=1,selected_epoch=0,selected_parameters={'prior':p},history=[],fit_seconds=0.,metrics={s:metrics([r['label'] for r in splits[s]],v) for s,v in predictions.items()},predictions=predictions)
    rng=np.random.default_rng(seed);q=pair_template();distribution=QuantumDistribution(q)
    size=4 if kind=='quantum' else 5 if kind=='linear' else 49
    params={('w' if kind=='quantum' else 'p')+str(i):float(v) for i,v in enumerate(rng.normal(0,.3,size))}
    opt=Adam(lr=lr);history=[];best=None;start=time.perf_counter()
    def evaluate(item,gradient=False):
        if kind=='quantum':
            bindings=dict(item['bindings'],**params)
            if gradient:
                value,jac=distribution.value_and_grad(bindings,wrt=list(params))
                return float(value[0]),{k:float(v[0]) for k,v in jac.items()}
            return float(distribution(bindings)[0])
        value,g=classical_value_grad(item['classical'],params,kind)
        return (value,g) if gradient else value
    for epoch in range(epochs+1):
        losses={split:float(np.mean([(evaluate(item)-item['label'])**2 for item in splits[split]]))
                for split in ('train','validation')}
        record=dict(epoch=epoch,train_loss=losses['train'],validation_loss=losses['validation'],parameters=dict(params))
        history.append(record)
        if best is None or record['validation_loss']<best['validation_loss']:best=record
        if epoch==epochs:break
        gradients=dict.fromkeys(params,0.)
        for item in splits['train']:
            value,g=evaluate(item,True)
            for k in params:gradients[k]+=2*(value-item['label'])*g[k]/len(splits['train'])
        params=opt.step(params,gradients)
    seconds=time.perf_counter()-start
    params=dict(best['parameters']) # validation-only checkpoint selection
    predictions={split:[evaluate(item) for item in items] for split,items in splits.items()}
    return dict(kind=kind,seed=seed,parameters=len(params),selected_epoch=best['epoch'],
                selected_parameters=params,history=history,fit_seconds=seconds,
                metrics={s:metrics([i['label'] for i in splits[s]],pred) for s,pred in predictions.items()},
                predictions=predictions)


def run(dataset,output,epochs=30,seeds=(7,13,21),lr=.05,feature_options=None,embeddings=None,include_majority=False):
    if epochs<1 or len(seeds)!=len(set(seeds)) or not seeds:raise ValueError('Use positive epochs and distinct seeds')
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    rows=load_pairs(dataset)
    training_texts=list(dict.fromkeys(r[k] for r in rows if r['split']=='train' for k in ('left','right')))
    if embeddings is not None:embeddings=dict(embeddings,fit_sentences=training_texts)
    features=SentenceFeatures(2,fit_sentences=training_texts,embeddings=embeddings,**(feature_options or {}))
    splits={s:[] for s in ('train','validation','test')}
    for row in rows:
        a,b=[features.transform(row[k]) for k in ('left','right')]
        item=dict(row,bindings=dict(zip(['a0','a1','b0','b1'],a['angles']+b['angles'])),
                  classical=classical_features(np.array(a['angles'])/np.pi,np.array(b['angles'])/np.pi),
                  unknown_words=sorted(set(a['unknown_words']+b['unknown_words'])))
        splits[row['split']].append(item)
    results=[]
    for seed in seeds:
        for kind in (('quantum','linear','mlp','majority') if include_majority else ('quantum','linear','mlp')):
            result=fit_model(kind,splits,seed,epochs,lr);results.append(result)
            print(kind,seed,'epoch',result['selected_epoch'],'test',result['metrics']['test'],flush=True)
    # Fixed first seed for diagnostic examples; never select a seed using test performance.
    quantum=results[0];q=pair_template();test=splits['test'];diagnostic=next((i for i in test if i['category']=='word_order'),test[0])
    other=next((i for i in test if i['category']=='paraphrase'),test[-1])
    checkpoints=quantum['history'][::max(1,epochs//4)]
    if checkpoints[-1]['epoch']!=epochs:checkpoints.append(quantum['history'][-1])
    checkpoints=checkpoints[:31]
    q.explain(bindings=dict(diagnostic['bindings'],**quantum['selected_parameters']),observable='P0',sensitivities=True,
        title='Trained sentence pairs: two held-out examples',
        parameter_roles={name:'weight' if name.startswith('w') else 'feature' for name in q.parameters},
        comparisons=[dict(name='Held-out paraphrase',bindings=dict(other['bindings'],**quantum['selected_parameters']))],
        training_history=[dict(epoch=r['epoch'],bindings=dict(diagnostic['bindings'],**r['parameters']),loss=r['train_loss']) for r in checkpoints]
    ).to_html(filename=output/'training.html')
    preprocessing=features.metadata()
    errors=[]
    for result in results:
        for item,score in zip(test,result['predictions']['test']):
            errors.append({k:item[k] for k in ('left','right','label','category','unknown_words')}|
                          dict(model=result['kind'],seed=result['seed'],score=score,correct=bool((score>=.5)==item['label']),error_groups=error_groups(item)))
    data=dict(dataset_sha256=hashlib.sha256(Path(dataset).read_bytes()).hexdigest(),epochs=epochs,seeds=list(seeds),
              learning_rate=lr,device='NumPy CPU complex128',
              environment=dict(python=sys.version,numpy=np.__version__,platform=platform.platform()),
              preprocessing=preprocessing,
              split_counts={s:len(v) for s,v in splits.items()},results=results,held_out=errors,
              protocol='Full-batch MSE, Adam, same split/features/update budget; checkpoint selected by validation MSE; threshold fixed at 0.5. Parameter budgets differ. Runtime is training plus validation, not a simulator benchmark.',
              limitations=['Inspect dataset provenance: fixture results are educational; public-data subsets are not official benchmark submissions.',
                           'Pair-disjoint splits are required; consult preparation metadata for sentence-disjointness.',
                           'Two-component reduction can lose word-order/context information even with bigrams or embeddings.',
                           'No hyperparameter tuning, confidence intervals over datasets or quantum advantage claim.'])
    (output/'results.json').write_text(json.dumps(data,indent=2,allow_nan=False))
    with (output/'held_out.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=errors[0]);writer.writeheader();writer.writerows(errors)
    render_report(data,output)
    return data


def error_groups(item):
    from collections import Counter
    a=SentenceFeatures.tokens(item['left']);b=SentenceFeatures.tokens(item['right'])
    neg={'not','no','never'};groups=[item['category']]
    if bool(set(a)&neg)!=bool(set(b)&neg):groups.append('negation_marker_mismatch')
    if a!=b and Counter(a)==Counter(b):groups.append('same_tokens_different_order')
    if item['unknown_words']:groups.append('unknown_words')
    return groups


def render_report(data,output):
    esc=html.escape
    content='<h1>Trained sentence-pair experiment</h1><p>'+esc(data['protocol'])+'</p><p>'+esc(str(data['split_counts']))+'</p>'
    content+='<table><tr><th>Model</th><th>Seed</th><th>Parameters</th><th>Selected epoch</th><th>Test MSE</th><th>Accuracy</th><th>F1</th><th>Fit seconds</th></tr>'
    for r in data['results']:
        m=r['metrics']['test'];content+='<tr>'+''.join('<td>'+esc(str(v))+'</td>' for v in [r['kind'],r['seed'],r['parameters'],r['selected_epoch'],round(m['mse'],5),round(m['accuracy'],4),round(m['f1'],4),round(r['fit_seconds'],3)])+'</tr>'
    content+='</table><h2>Mean held-out scores across declared seeds</h2><table><tr><th>Model</th><th>Mean MSE</th><th>Mean accuracy</th><th>Mean F1</th></tr>'
    for kind in dict.fromkeys(r['kind'] for r in data['results']):
        group=[r for r in data['results'] if r['kind']==kind]
        content+='<tr><td>'+kind+'</td>'+''.join('<td>'+str(round(float(np.mean([r['metrics']['test'][metric] for r in group])),5))+'</td>' for metric in ('mse','accuracy','f1'))+'</tr>'
    content+='</table><h2>Error groups (overlapping heuristics, not gold syntax labels)</h2><table><tr><th>Model</th><th>Group</th><th>Cases across seeds</th><th>Errors</th></tr>'
    for kind in dict.fromkeys(r['model'] for r in data['held_out']):
        for group in sorted({g for r in data['held_out'] for g in r['error_groups']}):
            cases=[r for r in data['held_out'] if r['model']==kind and group in r['error_groups']]
            content+='<tr>'+''.join('<td>'+esc(str(v))+'</td>' for v in [kind,group,len(cases),sum(not r['correct'] for r in cases)])+'</tr>'
    content+='</table><h2>Held-out errors and successes</h2><table><tr><th>Model / seed</th><th>Category</th><th>Left</th><th>Right</th><th>Label / score</th><th>Correct?</th></tr>'
    for r in data['held_out']:
        content+='<tr>'+''.join('<td>'+esc(str(v))+'</td>' for v in [r['model']+' / '+str(r['seed']),r['category'],r['left'],r['right'],str(r['label'])+' / '+str(round(r['score'],4)),r['correct']])+'</tr>'
    content+='</table><h2>Training playback</h2><p>First declared seed, validation-selected checkpoint. Inspect preprocessing when diagnosing errors; two-component compression can erase distinctions. Recorded training loss is over the full training set, not this one example.</p><iframe title="Training checkpoints" src="training.html"></iframe><h2>Limitations</h2><ul>'+''.join('<li>'+esc(s)+'</li>' for s in data['limitations'])+'</ul>'
    (output/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>QNLP training results</title><style>body{font:15px/1.5 system-ui;max-width:1200px;margin:30px auto;padding:16px;color:#20314b}table{border-collapse:collapse;width:100%;font-size:12px}th,td{padding:8px;border-bottom:1px solid #ddd;text-align:left}iframe{width:100%;height:1200px;border:1px solid #ddd}</style>'+content)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=Path(__file__).parent/'data/sentence_pairs.csv')
    parser.add_argument('--output',type=Path,default=Path('.release-artifacts/qnlp-training'))
    parser.add_argument('--epochs',type=int,default=30);parser.add_argument('--seeds',type=int,nargs='+',default=[7,13,21]);parser.add_argument('--lr',type=float,default=.05)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    run(args.data,args.output,args.epochs,tuple(args.seeds),args.lr);print('Open',(args.output/'report.html').resolve())


if __name__=='__main__':main()
