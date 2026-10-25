"""Offline NLP circuit lesson with supplied features; no parser/model download.

python examples/nlp_building_blocks.py --output .release-artifacts/nlp
"""
import argparse
from pathlib import Path
import numpy as np
from nexqus import nlp


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('.release-artifacts/nlp'))
    parser.add_argument('--device',default='cpu',help='cpu, cuda or cuda:N')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    # Deliberately supplied numbers, not embeddings learned from these words.
    training_features=np.array([[0.,0.,0.],[1.,1.,1.],[.4,.6,.2]])
    scaler=nlp.FeatureAngleScaler().fit(training_features)
    angles=scaler.transform([[.2,.6,.4],[.3,.5,.4]])
    left,right=[nlp.sentence_encoder(row) for row in angles]
    print('Supplied-feature state fidelity:',nlp.pair_fidelity(left,right,device=args.device))
    print('This overlap is not a measured semantic-similarity benchmark.')
    pair=nlp.sentence_pair_circuit(left,right)
    pair.explain(title='Sentence-pair overlap: supplied numeric features',device=args.device,
        blocks=[dict(name='Left encoder',role='encoding',start=0,stop=len(left.operations)),
                dict(name='Undo right encoder',role='readout',start=len(left.operations),stop=len(pair.operations))]
    ).to_html(filename=args.output/'sentence_pair.html')
    syntax=nlp.DependencyParse(('Alice','likes','Bob'),(1,1,1))
    circuit=syntax.circuit([.2,.6,.4])
    blocks=[dict(name=f'Encode token: {t}',role='encoding',start=i,stop=i+1)
            for i,t in enumerate(syntax.tokens)]
    blocks.append(dict(name='Dependency edges: likes → Alice, likes → Bob',
        role='entangling',start=3,stop=5))
    circuit.explain(title='Syntax-guided circuit: Alice likes Bob',blocks=blocks,
        device=args.device).to_html(filename=args.output/'dependency.html')
    print('Word/wire mapping:',syntax.annotations())
    print('Open reports in:',args.output.resolve())


if __name__=='__main__': main()
