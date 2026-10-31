"""Controlled feature ablation on a prepared labelled sentence-pair CSV."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from train_sentence_pairs import run,load_pairs
from nlp_similarity import SentenceFeatures


def study(data,output,epochs=10,seeds=(7,13,21),representations=('unigram','bigram'),embeddings=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=True);summary=[]
    rows=load_pairs(data);fit=list(dict.fromkeys(r[k] for r in rows if r['split']=='train' for k in ['left','right']))
    for name in representations:
        if name not in ('unigram','bigram','embeddings'):raise ValueError('Unknown representation')
        if name=='embeddings' and embeddings is None:raise ValueError('Supply a local embedding-vector JSON file')
        options=dict(ngrams=2 if name=='bigram' else 1,max_features=512)
        vectors=embeddings if name=='embeddings' else None
        result=run(data,output/name,epochs=epochs,seeds=seeds,feature_options=options,embeddings=vectors,include_majority=True)
        # Retention probe uses a training sentence, never test labels.
        original=fit[0];reversed_text=' '.join(reversed(SentenceFeatures.tokens(original)))
        feature=SentenceFeatures(2,fit_sentences=fit,**options) if vectors is None else None
        probe=None
        if feature is not None:
            a,b=feature.transform(original),feature.transform(reversed_text)
            probe=dict(source='First training sentence versus its token-order reversal (not a semantic label)',
                full_feature_l2=float(np.linalg.norm(np.array(a['features'])-b['features'])),
                reduced_l2=float(np.linalg.norm(np.array(a['reduced'])-b['reduced'])),
                angle_l2=float(np.linalg.norm(np.array(a['angles'])-b['angles'])))
        for model in dict.fromkeys(r['kind'] for r in result['results']):
            selected=[r for r in result['results'] if r['kind']==model]
            summary.append(dict(representation=name,model=model,
                mean_test_mse=float(np.mean([r['metrics']['test']['mse'] for r in selected])),
                mean_test_accuracy=float(np.mean([r['metrics']['test']['accuracy'] for r in selected])),
                mean_test_f1=float(np.mean([r['metrics']['test']['f1'] for r in selected])),
                seeds=list(seeds),probe=probe))
    record=dict(csv_sha256=hashlib.sha256(Path(data).read_bytes()).hexdigest(),epochs=epochs,results=summary,
        protocol='Same CSV, split, two circuit components, model architecture, Adam rate, seed list and update budget; only feature representation changes. No selection between representations using test scores.')
    metadata=Path(data).with_suffix('.metadata.json')
    if metadata.exists():record['preparation']=json.loads(metadata.read_text())
    (output/'comparison.json').write_text(json.dumps(record,indent=2))
    import html
    body='<h1>Controlled QNLP feature comparison</h1><p>'+html.escape(record['protocol'])+'</p><table><tr><th>Representation</th><th>Model</th><th>Mean test MSE</th><th>Accuracy</th><th>F1</th></tr>'
    for r in summary:body+='<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in [r['representation'],r['model'],round(r['mean_test_mse'],5),round(r['mean_test_accuracy'],4),round(r['mean_test_f1'],4)])+'</tr>'
    body+='</table><p>Majority uses the training label prior, with a fixed 0.5 threshold. The subsets are not official GLUE scores. Seeds are repeated fits of the same data, not independent test datasets.</p><ul>'
    for name in representations:body+='<li><a href="'+name+'/report.html">'+name+' losses, per-pair errors and training playback</a></li>'
    body+='</ul><h2>Order-retention probes</h2><pre>'+html.escape(json.dumps([r for r in summary if r['model']=='quantum'],indent=2))+'</pre>'
    (output/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>QNLP feature study</title><style>body{font:16px/1.5 system-ui;max-width:1000px;margin:30px auto;padding:16px}table{border-collapse:collapse;width:100%}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}pre{white-space:pre-wrap}</style>'+body)
    return record


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,default=Path('.release-artifacts/qnlp-features'))
    p.add_argument('--epochs',type=int,default=10);p.add_argument('--seeds',type=int,nargs='+',default=[7,13,21]);p.add_argument('--representations',nargs='+',choices=['unigram','bigram','embeddings'],default=['unigram','bigram']);p.add_argument('--embeddings',type=Path)
    a=p.parse_args();vectors=json.loads(a.embeddings.read_text()) if a.embeddings else None
    study(a.data,a.output,a.epochs,tuple(a.seeds),tuple(a.representations),vectors)
    print('Open',(a.output/'report.html').resolve())
