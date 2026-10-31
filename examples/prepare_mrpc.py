"""Explicit opt-in MRPC download and deterministic sentence-disjoint CSV preparation.

Parquet reading needs optional pyarrow. Training the prepared CSV needs only NumPy.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import urllib.request
import numpy as np

REVISION='68d3d763fc36f89a8d047f4b8e51b98ea36b98d0'
BASE=f'https://huggingface.co/datasets/nyu-mll/glue/resolve/{REVISION}/mrpc'
EXPECTED={'train':'61fd41301e0e244b0420c4350a170c8e7cf64740335fc875a4af2d79af0df0af',
          'validation':'33c007dbf5bfa8463d87a13e6226df8c0fcf2596c2cd39d0f3bb79754e00f50f'}


def normalize(s):return ' '.join(s.casefold().split())
def key(r):return tuple(sorted([normalize(r['left']),normalize(r['right'])]))
def sentences(rows):return {normalize(r[k]) for r in rows for k in ['left','right']}


def stratified(rows,size,rng):
    if isinstance(size,bool) or not isinstance(size,int) or size<2 or size>len(rows):raise ValueError('Subset size is unavailable or below two')
    groups=[[r for r in rows if r['label']==label] for label in [0,1]]
    if not all(groups):raise ValueError('Both labels are required')
    n0=max(1,min(size-1,round(size*len(groups[0])/len(rows))))
    counts=[n0,size-n0];chosen=[]
    for group,n in zip(groups,counts):
        if n>len(group):raise ValueError('Not enough examples of a required label')
        chosen.extend(group[i] for i in rng.permutation(len(group))[:n])
    return [chosen[i] for i in rng.permutation(len(chosen))]


def clean(rows):
    groups={}
    for r in rows:
        if r['label'] not in (0,1):raise ValueError('Unlabelled data cannot be evaluated')
        groups.setdefault(key(r),[]).append(r)
    result=[];duplicates=conflicts=0
    for group in groups.values():
        if len({r['label'] for r in group})>1:conflicts+=len(group);continue
        result.append(group[0]);duplicates+=len(group)-1
    return result,dict(duplicate_rows_removed=duplicates,conflicting_label_rows_removed=conflicts)


def split_rows(training,heldout,sizes=(256,64,128),seed=17):
    training,a=clean(training);heldout,b=clean(heldout);rng=np.random.default_rng(seed)
    test=stratified(heldout,sizes[2],rng);reserved=sentences(test)
    pool=[r for r in training if not (sentences([r])&reserved)]
    validation=stratified(pool,sizes[1],rng);reserved|=sentences(validation)
    candidates=[r for r in pool if not (sentences([r])&reserved)]
    train=stratified(candidates,sizes[0],rng)
    splits=dict(train=train,validation=validation,test=test)
    for x,y in [('train','validation'),('train','test'),('validation','test')]:
        if sentences(splits[x])&sentences(splits[y]):raise RuntimeError('Sentence leakage')
    return splits,dict(train_source_cleanup=a,heldout_source_cleanup=b,
        training_pool_after_test_exclusion=len(pool),training_pool_after_validation_exclusion=len(candidates),
        sentence_disjoint=True,seed=seed)


def prepare(cache,output,sizes=(256,64,128),seed=17,download=False):
    import pyarrow.parquet as pq
    cache=Path(cache);cache.mkdir(parents=True,exist_ok=True);sources={};raw={}
    for split in ['train','validation']:
        path=cache/(split+'.parquet');url=f'{BASE}/{split}-00000-of-00001.parquet'
        if not path.exists():
            if not download:raise ValueError('Missing cache: pass --download to fetch the pinned source')
            with urllib.request.urlopen(url,timeout=60) as response:path.write_bytes(response.read())
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        if digest!=EXPECTED[split]:raise ValueError('Source checksum mismatch: '+str(path))
        sources[split]=dict(url=url,sha256=digest)
        raw[split]=[dict(left=r['sentence1'],right=r['sentence2'],label=int(r['label']),category='paraphrase' if r['label']==1 else 'non_paraphrase') for r in pq.read_table(path).to_pylist()]
    splits,audit=split_rows(raw['train'],raw['validation'],sizes,seed)
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=['split','left','right','label','category']);writer.writeheader()
        for split,rows in splits.items():
            for row in rows:writer.writerow(dict(split=split,**row))
    metadata=dict(dataset='MRPC via NYU GLUE mirror',revision=REVISION,sources=sources,audit=audit,
        source_counts={s:len(r) for s,r in raw.items()},split_counts={s:len(r) for s,r in splits.items()},
        class_counts={s:{str(label):sum(r['label']==label for r in rows) for label in [0,1]} for s,rows in splits.items()},
        csv_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
        protocol='Test sampled from labelled GLUE validation; train and validation derived from GLUE train. Custom sentence-disjoint subsets, not official GLUE benchmark scores.',
        source_homepage='https://www.microsoft.com/en-us/download/details.aspx?id=52398')
    output.with_suffix('.metadata.json').write_text(json.dumps(metadata,indent=2));return metadata


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cache',type=Path,default=Path('.release-artifacts/mrpc-cache'));p.add_argument('--output',type=Path,default=Path('.release-artifacts/mrpc.csv'))
    p.add_argument('--sizes',type=int,nargs=3,default=[256,64,128]);p.add_argument('--seed',type=int,default=17);p.add_argument('--download',action='store_true');a=p.parse_args()
    print(json.dumps(prepare(a.cache,a.output,tuple(a.sizes),a.seed,a.download),indent=2))
