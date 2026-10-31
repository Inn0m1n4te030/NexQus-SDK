"""Local sentence-pair explorer. No network calls, accounts or text-model downloads.

Run from a development checkout: python examples/nlp_similarity.py
Use --export PATH for a standalone report with precomputed sample choices.
"""
import argparse
import json
from pathlib import Path
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import webbrowser
import numpy as np
from nexqus import nlp, Parameter

CORPUS = [
    'the cat sleeps on the sofa', 'the dog sleeps on the floor',
    'a cat rests beside the sofa', 'a dog runs in the park',
    'students study quantum circuits', 'students learn quantum computing',
    'a quantum circuit encodes data', 'classical computers process data',
    'the scientist studies molecules', 'the scientist builds a computer',
    'cats and dogs play outside', 'the student reads a science book',
]
SAMPLES = [
    ['the cat sleeps on the sofa', 'a cat rests beside the sofa',
     'students study quantum circuits', 'students learn quantum computing'],
    ['the cat sleeps on the sofa', 'the cat sleeps on the sofa',
     'the cat sleeps on the sofa', 'classical computers process data'],
    ['students learn quantum computing', 'students study quantum circuits',
     'the scientist studies molecules', 'the scientist builds a computer'],
]


def cosine(a, b):
    denominator=np.linalg.norm(a)*np.linalg.norm(b)
    return None if denominator < 1e-14 else float(np.clip(np.dot(a,b)/denominator,-1,1))


class SentenceFeatures:
    """Example-only train-fitted TF-IDF/SVD, or external sentence-vector lookup."""
    def __init__(self, qubits=3, embeddings=None, fit_sentences=None):
        if isinstance(qubits,bool) or not isinstance(qubits,int) or not 1<=qubits<=6:
            raise ValueError('qubits must be an integer from 1 to 6')
        self.lookup=None
        if embeddings is None:
            self.fit_sentences=list(CORPUS if fit_sentences is None else fit_sentences)
            if len(set(self.fit_sentences))<4 or not all(isinstance(s,str) and s.strip() for s in self.fit_sentences):
                raise ValueError('Provide at least four distinct fit sentences')
            tokens=[self.tokens(s) for s in self.fit_sentences]
            self.vocabulary=sorted({t for row in tokens for t in row})
            self.idf=np.log((1+len(tokens))/(1+np.array([sum(t in row for row in tokens) for t in self.vocabulary])))+1
            matrix=np.array([self._tfidf(row) for row in tokens])
            self.description='Offline word TF-IDF, L2 normalization, training-fitted truncated SVD'
        else:
            if not isinstance(embeddings,dict) or set(embeddings)!={'fit_sentences','vectors'}:
                raise ValueError('Embedding JSON needs fit_sentences and vectors')
            self.fit_sentences=embeddings['fit_sentences'];self.lookup=embeddings['vectors']
            if (not isinstance(self.fit_sentences,list) or len(self.fit_sentences)<4
                    or not all(isinstance(s,str) and s.strip() for s in self.fit_sentences)
                    or len(set(self.fit_sentences))!=len(self.fit_sentences) or not isinstance(self.lookup,dict)):
                raise ValueError('Provide at least four distinct fit sentences and a vectors mapping')
            vectors=list(self.lookup.values());raw=np.asarray(vectors)
            if raw.ndim!=2 or raw.dtype.kind not in 'iuf' or not np.isfinite(raw).all():
                raise ValueError('External vectors must be finite, real and fixed-width')
            if any(not isinstance(s,str) or not s.strip() for s in self.lookup):
                raise ValueError('Vector keys must be nonempty sentences')
            try:matrix=np.asarray([self.lookup[s] for s in self.fit_sentences],dtype=float)
            except KeyError as exc:raise ValueError('Every fit sentence needs a vector') from exc
            self.vocabulary=[];self.idf=None
            self.description='Externally supplied sentence vectors, training-fitted truncated SVD (no model loaded)'
        if qubits>min(matrix.shape):raise ValueError('Not enough fit rows/features for requested qubits')
        self.qubits=qubits
        _,_,vt=np.linalg.svd(matrix,full_matrices=False);self.projection=vt[:qubits].copy()
        # Stable display orientation; sign choice carries no linguistic meaning.
        for row in self.projection:
            if row[np.argmax(abs(row))]<0:row*=-1
        projected=matrix@self.projection.T
        self.minimum=projected.min(axis=0);self.maximum=projected.max(axis=0)
        self.scaler=nlp.FeatureAngleScaler().fit(projected)

    @staticmethod
    def tokens(sentence):return re.findall(r'[^\W_]+',sentence.casefold(),flags=re.UNICODE)

    def _tfidf(self,tokens):
        vector=np.array([tokens.count(t) for t in self.vocabulary],dtype=float)*self.idf
        norm=np.linalg.norm(vector)
        return vector/norm if norm else vector

    def transform(self,sentence):
        if not isinstance(sentence,str) or not sentence.strip() or len(sentence)>1000:
            raise ValueError('Each sentence must contain 1-1000 characters')
        tokens=self.tokens(sentence)
        if self.lookup is None:
            raw=self._tfidf(tokens);unknown=sorted(set(tokens)-set(self.vocabulary))
            if not np.any(raw):raise ValueError('No known words: use the displayed fit vocabulary, or supply external vectors')
        else:
            if sentence not in self.lookup:raise ValueError('Sentence has no external vector; add it to the vectors JSON')
            raw=np.asarray(self.lookup[sentence],dtype=float);unknown=[]
        reduced=raw@self.projection.T;angles=self.scaler.transform([reduced])[0]
        return dict(sentence=sentence,tokens=tokens,unknown_words=unknown,features=raw.tolist(),
                    reduced=reduced.tolist(),angles=angles.tolist(),
                    clipped=((reduced<self.minimum)|(reduced>self.maximum)).tolist())

    def metadata(self):
        return dict(description=self.description,fit_sentences=self.fit_sentences,
                    vocabulary=self.vocabulary,minimum=self.minimum.tolist(),maximum=self.maximum.tolist(),
                    projection=self.projection.tolist(),qubits=self.qubits)


class SimilarityDemo:
    def __init__(self,qubits=3,device='cpu',embeddings=None):
        self.features=SentenceFeatures(qubits,embeddings);self.device=device
        n=self.features.qubits
        self.names=[f'{side}{i}' for side in ['a','b'] for i in range(n)]
        left=nlp.sentence_encoder([Parameter(f'a{i}') for i in range(n)])
        right=nlp.sentence_encoder([Parameter(f'b{i}') for i in range(n)])
        self.circuit=nlp.sentence_pair_circuit(left,right)
        self.blocks=[dict(name='Left sentence encoder',start=0,stop=len(left.operations),role='encoding'),
                     dict(name='Inverse right encoder',start=len(left.operations),stop=len(self.circuit.operations),role='readout')]

    def compare(self,sentences):
        if not isinstance(sentences,list) or len(sentences)!=4:
            raise ValueError('Provide exactly four sentences: left/right for pair A and pair B')
        features=[self.features.transform(s) for s in sentences]
        bindings=[];pairs=[]
        for i in [0,2]:
            left,right=features[i:i+2]
            bindings.append(dict(zip(self.names,left['angles']+right['angles'])))
            a=nlp.sentence_encoder(left['angles']);b=nlp.sentence_encoder(right['angles'])
            pairs.append(dict(classical_cosine=cosine(np.array(left['features']),np.array(right['features'])),
                              reduced_cosine=cosine(np.array(left['reduced']),np.array(right['reduced'])),
                              fidelity=nlp.pair_fidelity(a,b,device=self.device)))
        report=self.circuit.explain(bindings=bindings[0],observable='P0',device=self.device,
            title='Sentence-pair fidelity: pair A versus pair B',blocks=self.blocks,
            parameter_roles=dict.fromkeys(self.names,'feature'),sensitivities=True,
            comparisons=[dict(name='Pair B',bindings=bindings[1])])
        # A second evaluation guards the UI's score/readout contract.
        traces=report.to_dict()
        for pair,trace in zip(pairs,[traces,traces['comparisons'][0]['trace']]):
            if not np.isclose(pair['fidelity'],trace['frames'][-1]['expectation'],atol=1e-10):
                raise RuntimeError('Fidelity readout does not match circuit explanation')
        return dict(sentences=features,pairs=pairs,preprocessing=self.features.metadata(),report=str(report.to_html()))

    def samples(self):
        if self.features.lookup is None:return SAMPLES
        s=self.features.fit_sentences
        return [[s[0],s[1],s[2],s[3]],[s[0],s[0],s[0],s[1]]]


HTML='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>NexQus sentence-pair laboratory</title>
<style>body{font:16px/1.55 system-ui;margin:0;background:#f3f5fa;color:#22283e}main{max-width:1200px;margin:auto;padding:28px}h1{font-size:34px;margin:5px 0}h2{font-size:21px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}.card{background:white;border:1px solid #dbe0ed;border-radius:12px;padding:18px;margin:18px 0}label{display:block}textarea{width:100%;box-sizing:border-box;min-height:70px;padding:10px;font:inherit}button,select{padding:10px;font:inherit;margin:8px 8px 0 0}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:8px;border-bottom:1px solid #ddd;text-align:left;vertical-align:top}.scroll{overflow:auto}.note{color:#526078}.warning{color:#935015}iframe{width:100%;height:1150px;border:1px solid #ddd}.error{color:#a22222}details{margin:15px 0}pre{white-space:pre-wrap;overflow-wrap:anywhere}@media(max-width:700px){.grid{grid-template-columns:1fr}main{padding:14px}}</style>
<main><p>NEXQUS / NLP LABORATORY</p><h1>Follow a sentence pair into a quantum circuit</h1><p>Compare two pairs using a classical cosine score and a quantum state-overlap score. Inspect every transformation and the actual circuit readout.</p>
<div class="card"><label for="sample">Offline examples</label><select id="sample"></select><div class="grid"><div><h2>Pair A</h2><label>First sentence<textarea id="s0" maxlength="1000"></textarea></label><label>Second sentence<textarea id="s1" maxlength="1000"></textarea></label></div><div><h2>Pair B</h2><label>First sentence<textarea id="s2" maxlength="1000"></textarea></label><label>Second sentence<textarea id="s3" maxlength="1000"></textarea></label></div></div><button id="run">Compare sentences</button><button id="download">Download analysis JSON</button><p id="mode" class="note"></p><p id="status" aria-live="polite"></p></div>
<div class="card"><h2>1. Compare scores</h2><div id="scores" class="scroll"></div><p class="note">Cosine is in [-1,1]; fidelity is in [0,1]. They measure different geometries. A higher number across these columns is not evidence that one method is more accurate. This small example has no held-out semantic benchmark or quantum advantage claim.</p></div>
<div class="card"><h2>2. Text → features → circuit angles</h2><p id="pipeline"></p><div id="features" class="scroll"></div><details><summary>Fit corpus, vocabulary and SVD projection</summary><pre id="fit"></pre></details><p class="note">The preprocessing is fitted once on the displayed corpus, never refitted on entered sentences. Word features ignore order and unknown words. SVD components are numeric mixtures, not named semantic concepts. Clipping and dimension reduction can make different sentences share a representation.</p></div>
<div class="card"><h2>3. Explain the actual similarity prediction</h2><p>The circuit runs U(right)† U(left). Its all-zero probability equals the squared overlap of the two encoded states. The explorer below uses that exact P0 readout; derivatives are with respect to encoded angles, not raw words.</p><p class="note">This fixed encoder has no trained weights. Its shared trailing entangler cancels in the overlap; do not interpret it as a learned language model. Training playback remains available for trainable models in the SDK's QNN example.</p><iframe id="report" title="Sentence-pair circuit explanation" sandbox="allow-scripts allow-downloads"></iframe></div></main>
<script id="initial" type="application/json">__DATA__</script><script>
'use strict';const samples=JSON.parse(document.getElementById('initial').textContent),$=id=>document.getElementById(id);let current=samples[0],busy=false;
function cellTable(root,headers,rows){root.replaceChildren();const table=document.createElement('table'),head=document.createElement('tr');for(const h of headers){const th=document.createElement('th');th.textContent=h;head.append(th);}table.append(head);for(const row of rows){const tr=document.createElement('tr');for(const value of row){const td=document.createElement('td');td.textContent=value;tr.append(td);}table.append(tr);}root.append(table);}
const fmt=v=>v===null?'undefined (zero vector)':Number(v).toFixed(6),vector=v=>v.map(fmt).join(', ');
function display(data){current=data;data.sentences.forEach((s,i)=>{$('s'+i).value=s.sentence;});cellTable($('scores'),['Pair','Cosine: full features','Cosine: reduced features','Quantum fidelity'],data.pairs.map((p,i)=>[i?'B':'A',fmt(p.classical_cosine),fmt(p.reduced_cosine),fmt(p.fidelity)]));
$('pipeline').textContent=data.preprocessing.description+' → '+data.preprocessing.qubits+' components → training-range scaling to [0, π] → RY angles.';
cellTable($('features'),['Sentence','Tokens','Unknown words','Full features','SVD components','Angles (radians)','Clipped components'],data.sentences.map((s,i)=>[(i<2?'A':'B')+(i%2+1),s.tokens.join(' '),s.unknown_words.join(', ')||'none',vector(s.features),vector(s.reduced),vector(s.angles),s.clipped.map((v,i)=>v?i:null).filter(v=>v!==null).join(', ')||'none']));$('fit').textContent=JSON.stringify(data.preprocessing,null,2);$('report').srcdoc=data.report;$('status').textContent='Computed scores and circuit trace loaded.';}
samples.forEach((s,i)=>{const option=document.createElement('option');option.value=i;option.textContent='Example '+(i+1);$('sample').append(option);});$('sample').onchange=()=>display(samples[Number($('sample').value)]);
const live=location.protocol==='http:'||location.protocol==='https:';for(let i=0;i<4;i++)$('s'+i).readOnly=!live;$('run').disabled=!live;$('mode').textContent=live?'Local server: edit sentences and compare. No text is sent to a cloud service.':'Offline snapshot: choose a calculated example. To enter new sentences, run python examples/nlp_similarity.py locally.';
$('run').onclick=async()=>{if(busy)return;busy=true;$('run').disabled=true;$('status').textContent='Calculating…';try{const response=await fetch('/api/compare',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({sentences:[0,1,2,3].map(i=>$('s'+i).value)})});const data=await response.json();if(!response.ok)throw Error(data.error||'Calculation failed');display(data);}catch(e){$('status').textContent='Calculation failed; displayed scores remain from the last successful calculation. '+e.message;}finally{busy=false;$('run').disabled=false;}};
$('download').onclick=()=>{const {report,...data}=current;const url=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='nexqus-sentence-similarity.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};display(current);
</script></html>'''


def render(samples):
    data=json.dumps(samples,ensure_ascii=False,allow_nan=False)
    for a,b in [('&',r'\u0026'),('<',r'\u003c'),('>',r'\u003e'),('\u2028',r'\u2028'),('\u2029',r'\u2029')]:data=data.replace(a,b)
    return HTML.replace('__DATA__',data)


def handler(demo,page):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def reply(self,status,body,mime):
            raw=body.encode('utf-8');self.send_response(status);self.send_header('Content-Type',mime+'; charset=utf-8')
            self.send_header('Content-Length',str(len(raw)));self.send_header('X-Content-Type-Options','nosniff');self.end_headers();self.wfile.write(raw)
        def do_GET(self):
            self.reply(200,page,'text/html') if self.path=='/' else self.reply(404,'Not found','text/plain')
        def do_POST(self):
            if self.path!='/api/compare':return self.reply(404,'Not found','text/plain')
            origin=self.headers.get('Origin')
            allowed={f'http://127.0.0.1:{self.server.server_port}',f'http://localhost:{self.server.server_port}'}
            if origin and origin not in allowed:return self.reply(403,'{"error":"Origin rejected"}','application/json')
            try:
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=16384:raise ValueError('Request size must be 1-16384 bytes')
                if self.headers.get_content_type()!='application/json':raise ValueError('Use application/json')
                request=json.loads(self.rfile.read(length))
                if not isinstance(request,dict) or set(request)!={'sentences'}:raise ValueError('Request needs sentences only')
                data=demo.compare(request['sentences'])
                self.reply(200,json.dumps(data,ensure_ascii=False,allow_nan=False),'application/json')
            except (ValueError,TypeError) as exc:self.reply(400,json.dumps(dict(error=str(exc))),'application/json')
    return Handler


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,default=8767);parser.add_argument('--qubits',type=int,default=3)
    parser.add_argument('--device',default='cpu',help='cpu, cuda:0 or auto')
    parser.add_argument('--embeddings',type=Path,help='Optional local JSON sentence-vector lookup; no model downloads')
    parser.add_argument('--no-browser',action='store_true');parser.add_argument('--export',type=Path,help='Write standalone HTML and exit')
    args=parser.parse_args();embeddings=json.loads(args.embeddings.read_text()) if args.embeddings else None
    demo=SimilarityDemo(args.qubits,args.device,embeddings)
    page=render([demo.compare(s) for s in demo.samples()])
    if args.export:
        args.export.parent.mkdir(parents=True,exist_ok=True);args.export.write_text(page,encoding='utf-8');print(args.export.resolve());return
    with ThreadingHTTPServer(('127.0.0.1',args.port),handler(demo,page)) as server:
        url=f'http://127.0.0.1:{server.server_port}';print('Open',url,flush=True)
        if not args.no_browser:webbrowser.open(url)
        try:server.serve_forever()
        except KeyboardInterrupt:pass


if __name__=='__main__':main()
