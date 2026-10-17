"""Native diagrams and logical inspection; PNG/PDF are optional source additions."""
import argparse
from nexqus import Circuit, algorithms


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', action='store_true',help='Also export PNG/PDF (visualization extra and Cairo)')
    args = parser.parse_args()
    q = algorithms.teleportation()
    print(q.draw())
    q.draw('svg', filename='teleportation.svg')
    print('Saved teleportation.svg')
    custom = Circuit(8).rx(0.5, 0).cx(0, 1).ry(0.8, 3).cx(3, 7)
    print(custom.draw())
    print('Logical depth:', custom.depth(), 'gate counts:', custom.count_ops(gates_only=True))
    custom.draw('svg', filename='custom_8_qubits.svg')
    print('Saved custom_8_qubits.svg')
    if args.binary:
        for kind in ('png','pdf'):
            custom.draw(kind,filename='custom_8_qubits.'+kind)
            print('Saved custom_8_qubits.'+kind)


if __name__ == '__main__':
    main()
