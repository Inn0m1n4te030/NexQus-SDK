"""Independent Grover identities, maze coverage and local HTTP integration."""
import http.client
import importlib.util
import json
from pathlib import Path
import threading
import unittest
from unittest.mock import patch
import numpy as np
from nexqus import Simulator, algorithms
from nexqus.maze import Maze, PRESETS, breadth_first, check_plan, explore
from nexqus.maze_html import render_explorer


class MazeTests(unittest.TestCase):
    def test_paths_and_canonical_choice_encoding(self):
        maze = Maze(PRESETS['Two corridors']); result = explore(maze)
        valid = [p for p in result.candidates if p['valid']]
        self.assertEqual([p['bits'] for p in valid], ['000000', '010000'])
        self.assertEqual(len(breadth_first(maze)['path'])-1, 6)
        for plan in valid:
            self.assertEqual(plan['path'][0], maze.start); self.assertEqual(plan['path'][-1], maze.goal)
            self.assertEqual(len(plan['path']), len(set(plan['path'])))
            for first, second in zip(plan['path'], plan['path'][1:]):
                self.assertIn(second, maze.neighbors(first))
        hallway = Maze(PRESETS['Long hallway'])
        self.assertEqual(len(breadth_first(hallway)['path'])-1, 16)
        self.assertTrue(check_plan(hallway, '000000')['valid'])
        padded = check_plan(hallway, '000001')
        self.assertTrue(padded['reached_goal']); self.assertFalse(padded['valid'])
        self.assertIn('unused', padded['reason'])

    def test_marking_amplification_and_independent_formula(self):
        maze = Maze(PRESETS['Two corridors']); result = explore(maze)
        marked = np.array([p['valid'] for p in result.candidates])
        theta = np.arcsin(np.sqrt(marked.sum()/len(marked)))
        for i, frame in enumerate(result.frames):
            state = np.array(frame['real'])+1j*np.array(frame['imag'])
            self.assertAlmostEqual(float(np.vdot(state, state).real), 1)
            np.testing.assert_allclose(frame['probabilities'], abs(state)**2, atol=1e-14)
            if frame['phase'] == 'mark':
                previous = result.frames[i-1]
                np.testing.assert_allclose(frame['probabilities'], previous['probabilities'], atol=1e-14)
                np.testing.assert_allclose(frame['real'], np.array(previous['real'])*np.where(marked,-1,1), atol=1e-14)
            if frame['phase'] == 'amplify':
                previous = np.array(result.frames[i-1]['real'])
                np.testing.assert_allclose(state.real, 2*previous.mean()-previous, atol=1e-14)
                self.assertAlmostEqual(frame['success_probability'], np.sin((2*frame['iteration']+1)*theta)**2)
        self.assertGreater(result.frames[-1]['success_probability'], .999)
        marked_bits = [p['bits'] for p in result.candidates if p['valid']]
        reference = Simulator().run(algorithms.grover(6, marked_bits, result.iterations), shots=0)
        np.testing.assert_allclose(result.frames[-1]['real'], reference.statevector.real, atol=1e-14)
        self.assertLess(explore(maze, iterations=8).frames[-1]['success_probability'], .1)

    def test_no_path_and_insufficient_plan_budget_are_different(self):
        blocked = Maze(PRESETS['Blocked exit']); challenge = Maze(PRESETS['Choice-budget challenge'])
        self.assertEqual(breadth_first(blocked)['path'], [])
        self.assertTrue(breadth_first(challenge)['path'])
        for maze in (blocked, challenge):
            experiment = explore(maze)
            self.assertFalse(any(p['valid'] for p in experiment.candidates))
            self.assertEqual(experiment.iterations,0); self.assertEqual(len(experiment.frames),1)
            self.assertEqual(experiment.frames[0]['success_probability'],0)

    def test_oracle_compiler_does_not_use_bfs_or_minimize_length(self):
        maze = Maze(('S.G', '...'))
        with patch('nexqus.maze.breadth_first', side_effect=AssertionError('BFS cannot mark the oracle')):
            experiment = explore(maze)
        lengths = {p['moves'] for p in experiment.candidates if p['valid']}
        self.assertIn(2,lengths); self.assertIn(4,lengths)

    def test_measurement_is_seeded_unfiltered_and_uses_current_stage(self):
        experiment = explore(Maze(PRESETS['Two corridors']))
        self.assertEqual(experiment.measure(seed=7),experiment.measure(seed=7))
        failures = [experiment.measure(frame=0, seed=seed) for seed in range(16)]
        self.assertTrue(any(not sample['valid'] for sample in failures))
        sample = experiment.measure(frame=0, seed=3)
        self.assertEqual(sample['frame'],0)
        self.assertEqual(sample['path'], experiment.candidates[sample['index']]['path'])
        self.assertEqual(len(experiment.to_dict()['samples']),len(experiment.frames))

    def test_limits_and_export_safety(self):
        for rows in ([], ['SG'], ['SS','G.'], ['Sx','.G'], ['S.','G'], ['S.']*9, 'SG'):
            with self.subTest(rows=rows),self.assertRaises(ValueError): Maze(rows)
        maze=Maze(PRESETS['Two corridors'])
        for options in (dict(junctions=0),dict(junctions=True),dict(junctions=4),dict(iterations=9),dict(iterations=-1),dict(iterations=.5)):
            with self.subTest(options=options),self.assertRaises(ValueError): explore(maze,**options)
        experiment=explore(maze)
        with self.assertRaises(ValueError): experiment.measure(frame=True)
        with self.assertRaises(ValueError): experiment.measure(seed=-1)
        with self.assertRaises(ValueError): check_plan(maze,'00000x')
        data=experiment.to_dict(offline=True)
        hostile='</script><img src=x onerror="alert(1)">'
        data['presets'][hostile]=data['rows']; report=render_explorer(data)
        self.assertNotIn(hostile,report); self.assertIn('\\u003c/script\\u003e',report)
        self.assertIn('maze_oracle',experiment.explanation())
        self.assertEqual(data['resources']['classical_route_checks'],64)


class MazeHTTPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec=importlib.util.spec_from_file_location('maze_example',Path(__file__).resolve().parents[1]/'examples'/'quantum_maze.py')
        cls.example=importlib.util.module_from_spec(spec);spec.loader.exec_module(cls.example)
        from http.server import ThreadingHTTPServer
        cls.server=ThreadingHTTPServer(('127.0.0.1',0),cls.example.handler_for(explore(Maze(PRESETS['Two corridors']))))
        cls.thread=threading.Thread(target=cls.server.serve_forever,daemon=True);cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown();cls.server.server_close();cls.thread.join(timeout=2)

    def request(self,method,path,body=None,headers=None):
        connection=http.client.HTTPConnection('127.0.0.1',self.server.server_port,timeout=10)
        connection.request(method,path,body=body,headers=headers or {})
        response=connection.getresponse();payload=response.read();status=response.status;connection.close()
        return status,payload

    def test_server_solve_measure_and_explain(self):
        status,page=self.request('GET','/');self.assertEqual(status,200);self.assertIn(b'Quantum Maze Explorer',page)
        payload=dict(rows=PRESETS['Two corridors'],junctions=3,iterations=0,seed=11,frame=0)
        for path in ('/api/solve','/api/measure','/api/explain'):
            with self.subTest(path=path):
                status,body=self.request('POST',path,json.dumps(payload),{'Content-Type':'application/json'})
                self.assertEqual(status,200);result=json.loads(body)
                if path.endswith('solve'):self.assertEqual(result['iterations'],0);self.assertEqual(result['seed'],11)
                elif path.endswith('measure'):self.assertEqual(result['frame'],0);self.assertEqual(result['seed'],11)
                else:self.assertIn('Quantum Maze Explorer', result['html'])

    def test_invalid_requests_and_origin_limits(self):
        status,_=self.request('POST','/api/solve','{}',{'Content-Type':'text/plain'});self.assertEqual(status,415)
        for body in ('{','[]',json.dumps(dict(rows=PRESETS['Two corridors'],junctions=100))):
            status,_=self.request('POST','/api/solve',body,{'Content-Type':'application/json'});self.assertEqual(status,400)
        status,_=self.request('POST','/api/solve','{}',{'Content-Type':'application/json','Origin':'https://example.com'});self.assertEqual(status,403)
        status,_=self.request('GET','/',headers={'Host':'example.com'});self.assertEqual(status,403)
        status,_=self.request('GET','/not-a-file');self.assertEqual(status,404)


if __name__=='__main__':unittest.main()
