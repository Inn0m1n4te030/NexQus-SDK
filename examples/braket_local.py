"""Offline plan by default; --local uses only Braket's local CPU simulator."""
import argparse
from nexqus import Circuit
from nexqus.braket import prepare_braket


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--local',action='store_true',help='Run the free local simulator; never use AWS')
    args=parser.parse_args()
    q=Circuit(8).rx(.5,0).cx(0,1).ry(.8,3).cx(3,7)
    plan=prepare_braket(q,shots=1000)
    print(q.draw())
    print('Braket gate counts (including idle-wire identities):',plan.gate_counts)
    print('Prepared offline. No AWS session or cloud task has been created.')
    if args.local:
        result=plan.run_local()
        print('Local CPU counts:',result.counts)
        print('Sampled ZZ expectation:',result.expectation('ZZIIIIII'))


if __name__=='__main__': main()
