"""Launch the local school-project maze lab, or export a self-contained lesson.

python -m pip install -e .
python examples/quantum_maze.py
python examples/quantum_maze.py --export maze_lesson.html

No Flask, GPU, cloud credentials or QPU required. Ctrl+C stops the server.
"""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from urllib.parse import urlsplit
import webbrowser
from nexqus.maze import Maze, PRESETS, explore
from nexqus.maze_html import render_explorer


def experiment_from(payload):
    if not isinstance(payload, dict): raise ValueError('Supply a JSON object')
    return explore(Maze(payload.get('rows')), payload.get('junctions', 3), payload.get('iterations'))


def handler_for(initial, seed=7):
    page = render_explorer(initial.to_dict(seed)).encode('utf-8')

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup(); self.connection.settimeout(10)

        def log_message(self, format, *args):
            pass  # Keep school-project terminals readable.

        def _local(self):
            allowed = {f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}'}
            if self.server.server_port == 80: allowed.update(('127.0.0.1', 'localhost'))
            if self.headers.get('Host') not in allowed: return False
            origin = self.headers.get('Origin')
            return origin is None or origin in {'http://'+host for host in allowed}

        def _send(self, status, payload, content_type='application/json; charset=utf-8'):
            if not isinstance(payload, bytes): payload = json.dumps(payload, allow_nan=False).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(payload)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('X-Frame-Options', 'DENY')
            self.send_header('Content-Security-Policy', "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'; img-src data:; frame-ancestors 'none'; base-uri 'none'")
            self.end_headers(); self.wfile.write(payload)

        def do_GET(self):
            if not self._local(): self._send(403, dict(error='Only local requests are accepted')); return
            path = urlsplit(self.path).path
            if path == '/': self._send(200, page, 'text/html; charset=utf-8')
            elif path == '/health': self._send(200, dict(status='ok', backend='NexQus NumPy CPU'))
            else: self._send(404, dict(error='Not found'))

        def do_POST(self):
            if not self._local(): self._send(403, dict(error='Only same-origin local requests are accepted')); return
            endpoint = urlsplit(self.path).path
            if endpoint not in ('/api/solve', '/api/measure', '/api/explain'):
                self._send(404, dict(error='Not found')); return
            if self.headers.get('Content-Type', '').split(';')[0].strip().lower() != 'application/json':
                self._send(415, dict(error='Use application/json')); return
            try:
                length = int(self.headers.get('Content-Length', '-1'))
                if not 0 < length <= 32768: self._send(413, dict(error='JSON body must be 1..32768 bytes')); return
                payload = json.loads(self.rfile.read(length))
                experiment = experiment_from(payload)
                seed = payload.get('seed', 7)
                if endpoint == '/api/solve': result = experiment.to_dict(seed)
                elif endpoint == '/api/measure': result = experiment.measure(payload.get('frame'), seed)
                else: result = dict(html=experiment.explanation())
            except (ValueError, TypeError, UnicodeError, RecursionError) as error:
                self._send(400, dict(error=str(error))); return
            self._send(200, result)

    return Handler


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preset', choices=list(PRESETS), default='Two corridors')
    parser.add_argument('--junctions', type=int, choices=(1, 2, 3), default=3)
    parser.add_argument('--iterations', type=int, choices=range(9), default=None)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--port', type=int, default=8765, help='Local port; use 0 to choose a free port')
    parser.add_argument('--no-browser', action='store_true')
    parser.add_argument('--export', type=Path, help='Write a self-contained offline lesson instead of serving')
    args = parser.parse_args(argv)
    if not 0 <= args.port <= 65535 or not 0 <= args.seed <= 2**32-1:
        parser.error('Port must be 0..65535 and seed 0..4294967295')
    experiment = explore(Maze(PRESETS[args.preset]), args.junctions, args.iterations)
    if args.export:
        args.export.parent.mkdir(parents=True, exist_ok=True)
        args.export.write_text(render_explorer(experiment.to_dict(args.seed, offline=True), experiment.explanation()), encoding='utf-8')
        print('Open offline lesson:', args.export.resolve()); return
    try: server = ThreadingHTTPServer(('127.0.0.1', args.port), handler_for(experiment, args.seed))
    except OSError as error: parser.error(f'{error}. Try --port 0 for a free local port.')
    url = f'http://127.0.0.1:{server.server_port}'
    print('Local CPU simulation; no QPU or cloud calls.', flush=True)
    print('Open '+url+' — press Ctrl+C to stop.', flush=True)
    if not args.no_browser: webbrowser.open(url)
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally: server.server_close()


if __name__ == '__main__': main()
