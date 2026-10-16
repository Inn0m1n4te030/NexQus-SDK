"""Native diagrams: run this script to print and save a teleportation circuit."""
from nexqus import Circuit, algorithms


def main():
    q = algorithms.teleportation()
    print(q.draw())
    q.draw('svg', filename='teleportation.svg')
    print('Saved teleportation.svg')
    custom = Circuit(8).rx(0.5, 0).cx(0, 1).ry(0.8, 3).cx(3, 7)
    print(custom.draw())
    custom.draw('svg', filename='custom_8_qubits.svg')
    print('Saved custom_8_qubits.svg')


if __name__ == '__main__':
    main()
