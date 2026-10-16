"""Native diagrams: run this script to print and save a teleportation circuit."""
from nexqus import algorithms


def main():
    q = algorithms.teleportation()
    print(q.draw())
    q.draw('svg', filename='teleportation.svg')
    print('Saved teleportation.svg')


if __name__ == '__main__':
    main()
