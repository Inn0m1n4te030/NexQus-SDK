"""Compare a monitored quantum walk, matched classical walk and BFS locally.

python examples/quantum_maze_walk.py --steps 32
python examples/quantum_maze_walk.py --preset "Open 6 by 6" --steps 64
python examples/quantum_maze_walk.py --maze my_maze.txt --steps 64
"""
import argparse
import json
from pathlib import Path
import webbrowser
from nexqus.maze import Maze
from nexqus.maze_walk import WALK_PRESETS, compare_walks
from nexqus.maze_walk_html import render_walk


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preset', choices=list(WALK_PRESETS), default='Two corridors')
    parser.add_argument('--maze', type=Path, help='Text grid with . # S G; 2..8 rows/columns')
    parser.add_argument('--steps', type=int, default=32, help='1..128 walk steps; chosen before comparison')
    parser.add_argument('--output', type=Path, default=Path('.release-artifacts/maze-walk'))
    parser.add_argument('--open', action='store_true', help='Open the saved report in your browser')
    args = parser.parse_args()
    try:
        rows = args.maze.read_text(encoding='utf-8').splitlines() if args.maze else WALK_PRESETS[args.preset]
        result = compare_walks(Maze(rows), args.steps)
    except (ValueError, TypeError, OSError) as error: parser.error(str(error))
    args.output.mkdir(parents=True, exist_ok=True)
    report = args.output/'walk.html'
    report.write_text(render_walk(result), encoding='utf-8')
    (args.output/'walk.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(f"Exit detected by step {args.steps}: quantum {result['quantum_hit']:.3%}; classical walk {result['classical_hit']:.3%}")
    print('BFS:', result['bfs']['expanded_nodes'], 'visits;',
          len(result['bfs']['path'])-1 if result['bfs']['path'] else 'no', 'path moves')
    print('Median CPU ms:', result['timings']['median_ms'])
    print('Walkers detect arrival only; they do not output the route. No BFS speedup established.')
    print('Open', report.resolve())
    if args.open: webbrowser.open(report.resolve().as_uri())


if __name__ == '__main__': main()
