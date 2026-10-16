"""Educational Shor factoring. Run after installing this source checkout."""
from nexqus import algorithms


def main():
    for modulus in (15, 21):
        result = algorithms.shor(modulus, base=2, seed=42)
        print(f'{modulus}: factors={result.factors}, method={result.method}, order={result.order}')
        if result.factors is None:
            print('No factors recovered; try another base or seed.')


if __name__ == '__main__':
    main()
