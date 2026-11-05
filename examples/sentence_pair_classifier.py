"""Train a small explainable NLP classifier on CPU; no network or QPU calls."""
import argparse
import html
import json
from pathlib import Path
import numpy as np
from nexqus import SentencePairClassifier
from nlp_similarity import SentenceFeatures
from train_sentence_pairs import load_pairs, metrics, fit_model, classical_features


def run(output,epochs):
    rows=load_pairs(Path(__file__).parent/'data/sentence_pairs.csv')
    texts=list(dict.fromkeys(r[k] for r in rows if r['split']=='train' for k in ('left','right')))
    features=SentenceFeatures(2,fit_sentences=texts,ngrams=2)
    splits={s:[r for r in rows if r['split']==s] for s in ('train','validation','test')}
    def encode(rows): return np.array([[features.transform(r[k])['angles'] for k in ('left','right')] for r in rows])
    x={s:encode(v) for s,v in splits.items()}; y={s:np.array([r['label'] for r in v]) for s,v in splits.items()}
    model=SentencePairClassifier().fit(x['train'],y['train'],epochs=epochs,validation_data=(x['validation'],y['validation']))
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    model.save(output/'model.json')
    restored=SentencePairClassifier.load(output/'model.json')
    np.testing.assert_allclose(model.predict_proba(x['test']),restored.predict_proba(x['test']))
    prepared={s:[dict(r,classical=classical_features(pair[0]/np.pi,pair[1]/np.pi)) for r,pair in zip(splits[s],x[s])] for s in splits}
    baseline=fit_model('linear',prepared,7,epochs,.05)
    results={'quantum':metrics(y['test'],model.predict_proba(x['test'])[:,1]),'classical_linear':baseline['metrics']['test'],
             'selected_epoch':model.selected_epoch_,'train_loss_before':model.history_[0]['loss'],'train_loss_after':model.history_[-1]['loss']}
    (output/'metrics.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    model.explain(x['test'][0],comparison=x['test'][1]).to_html(filename=output/'circuit.html')
    pair=splits['test'][0]; score=float(model.predict_proba(x['test'][:1])[0,1])
    escaped=lambda obj:html.escape(json.dumps(obj,ensure_ascii=False,indent=2))
    sections=''.join(f'<h3>{html.escape(k)}</h3><pre>{escaped(features.transform(pair[k]))}</pre>' for k in ('left','right'))
    page=f'''<!doctype html><meta charset="utf-8"><title>Explainable sentence-pair classifier</title>
<style>body{{font:17px system-ui;max-width:1200px;margin:32px auto;padding:20px}}pre{{white-space:pre-wrap;background:#eef2f7;padding:16px}}iframe{{width:100%;height:850px;border:1px solid #aaa}}</style>
<h1>From sentences to a quantum prediction</h1><p>သင်ယူထားသော model ၏ prediction ကို circuit အဆင့်ဆင့် ကြည့်ပါ။</p>
<p>Class 1: related/paraphrase; class 0: unrelated. Threshold: 0.5.</p>
<h2>1. Input sentences and encoded features</h2>{sections}
<p>Training-only word/bigram TF-IDF → normalized vectors → training-fitted SVD → angle scaling.
These classical preprocessing steps are external to the SDK. Unknown terms and feature details appear above.</p>
<h2>2. Prediction</h2><p>P0 = {score:.6f}; predicted class = {int(score>=.5)}; reference label = {pair['label']}.</p>
<p>The circuit applies U(right)† U(left). Its all-zero probability measures overlap. Shared trainable rotations occur between feature uploads.
Identical encodings always score one; scores are not calibrated confidence.</p>
<h2>3. Inspect the internal circuit</h2><p>Step through gates and states, inspect local angle derivatives, compare the second held-out pair,
and replay recorded training checkpoints. Derivatives concern encoded angles, not word importance.
The comparison changes the pair; it is not a causal explanation.</p><iframe title="Circuit explorer" src="circuit.html"></iframe>
<h2>4. Held-out classical comparison</h2><pre>{escaped(results)}</pre>
<p>Same training data, encoded features, update budget and validation selection; parameter counts differ.
Small fixture demonstration; broader evaluation remains outstanding.</p>'''
    (output/'report.html').write_text(page,encoding='utf-8')
    print(json.dumps(results,indent=2)); print('Open',output.resolve()/'report.html')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs',type=int,default=20)
    parser.add_argument('--output',default='.release-artifacts/sentence-pair-classifier')
    args=parser.parse_args();run(args.output,args.epochs)
