"""Reproduce continuous-time glued-tree transport with an exact column model.

python examples/glued_trees.py --open
python examples/glued_trees.py --depths 4 8 16 32 --open
Full graphs are verified at depths 2,3,4; larger graphs are NOT materialized.
"""
import argparse
import json
from pathlib import Path
import webbrowser
from nexqus.glued_trees import compare_depths
from nexqus.glued_trees_html import render_glued_trees


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--depths',type=int,nargs='+',default=[2,4,8,16,32])
    parser.add_argument('--samples',type=int,default=129,help='17..257 time points for inspection only')
    parser.add_argument('--seed',type=int,default=7,help='Random cycle and hidden names for full-graph checks')
    parser.add_argument('--output',type=Path,default=Path('.release-artifacts/glued-trees'))
    parser.add_argument('--open',action='store_true')
    args = parser.parse_args()
    try: result = compare_depths(args.depths,args.seed,args.samples)
    except (ValueError,TypeError) as error: parser.error(str(error))
    args.output.mkdir(parents=True,exist_ok=True)
    report = args.output/'glued_trees.html'
    report.write_text(render_glued_trees(result),encoding='utf-8')
    (args.output/'glued_trees.json').write_text(json.dumps(result,allow_nan=False,indent=2)+'\n',encoding='utf-8')
    for row in result['results']:
        print(f"depth {row['depth']:2}: implied vertices {row['vertices']:,}; simulated columns {row['columns']}; "
              f"uniform-time exit: quantum {100*row['uniform_quantum_exit']:.6g}%, classical {100*row['uniform_classical_exit']:.6g}%")
    print('Full-graph checks:',result['verification'])
    print('CPU column simulation, not a coherent hidden-name oracle or a QPU benchmark.')
    print('Open',report.resolve())
    if args.open: webbrowser.open(report.resolve().as_uri())


if __name__=='__main__': main()
