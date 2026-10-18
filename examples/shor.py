"""Educational Shor factoring. Run after installing this source checkout."""
import argparse
from nexqus import algorithms,Simulator


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',help='cpu, cuda, cuda:N or auto')
    args=parser.parse_args(); simulator=Simulator.for_device(args.device)
    print('Execution:',simulator.backend.name,simulator.backend.device)
    for modulus in (15, 21, 35, 77):
        resources = algorithms.shor_resources(modulus)
        print(f'{resources.total_qubits} qubits; '
              f'{resources.estimated_working_bytes / 1024**2:.1f} MiB estimated state working storage')
        result = algorithms.shor(modulus, base=2, seed=42,simulator=simulator)
        print(f'{modulus}: factors={result.factors}, method={result.method}, order={result.order}')
        if result.factors is None:
            print('No factors recovered; try another base or seed.')


if __name__ == '__main__':
    main()
