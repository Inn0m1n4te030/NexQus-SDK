"""Independent matrix evolution and Markov tests for the monitored walk."""
import json
import unittest
from unittest.mock import patch
import numpy as np
from nexqus import Simulator
from nexqus.maze import Maze, PRESETS
from nexqus.maze_walk import walk_operators, compare_walks, _quantum, _classical
from nexqus.maze_walk_html import render_walk


class MazeWalkTests(unittest.TestCase):
    def test_shift_is_reversible_local_and_does_not_enter_walls(self):
        for maze in (Maze(PRESETS['Two corridors']), Maze(('S..', '..G'))):
            circuit, initial, mapping, goal = walk_operators(maze)
            size = len(mapping); width = len(maze.rows[0])
            np.testing.assert_array_equal(np.sort(mapping), np.arange(size))
            np.testing.assert_array_equal(mapping[mapping], np.arange(size))
            for source, target in enumerate(mapping):
                if source == target: continue
                r,c = divmod(source//4, width); nr,nc = divmod(target//4,width)
                self.assertEqual(abs(r-nr)+abs(c-nc), 1)
                self.assertNotEqual(maze.rows[nr][nc], '#')
                self.assertEqual(target%4, (source%4+2)%4)
            result = Simulator().run(circuit, initial_state=initial, shots=0)
            self.assertAlmostEqual(float(abs(result.statevector).dot(abs(result.statevector))),1)

    def test_sdk_trajectory_matches_independent_dense_projected_unitary(self):
        maze = Maze(('S..', '.#G'))
        circuit, initial, mapping, goal = walk_operators(maze)
        size=len(initial); coin=np.kron(np.eye(size//4), np.full((4,4),.5)-np.eye(4))
        shift=np.zeros((size,size)); shift[mapping,np.arange(size)]=1
        unitary=shift@coin
        np.testing.assert_allclose(unitary.T@unitary,np.eye(size))
        state=initial.copy(); hit=0.; frames,_=_quantum(circuit,initial,goal,12,trace=True)
        for frame in frames:
            np.testing.assert_allclose(frame['real'],state.real,atol=1e-13)
            self.assertAlmostEqual(frame['cumulative_hit'],hit)
            self.assertAlmostEqual(sum(frame['probabilities'])+hit,1)
            state=unitary@state
            hit+=float(sum(abs(state[4*goal:4*goal+4])**2)); state[4*goal:4*goal+4]=0

    def test_classical_is_independent_grid_markov_chain(self):
        maze=Maze(PRESETS['Two corridors']); width=len(maze.rows[0]); count=len(maze.rows)*width
        _,initial,mapping,goal=walk_operators(maze)
        # Build a position-only matrix directly from geometry, independent of S.
        transition=np.zeros((count,count))
        for r,row in enumerate(maze.rows):
            for c,symbol in enumerate(row):
                if symbol=='#': continue
                source=r*width+c
                for dr,dc in ((0,1),(1,0),(0,-1),(-1,0)):
                    target=(r+dr,c+dc)
                    dest=target[0]*width+target[1] if target in maze.neighbors((r,c)) else source
                    transition[dest,source]+=.25
        state=np.zeros(count); state[maze.start[0]*width+maze.start[1]]=1; hit=0.
        frames,_=_classical(initial,mapping,goal,16,trace=True)
        for frame in frames:
            np.testing.assert_allclose(frame['probabilities'],state)
            self.assertAlmostEqual(frame['cumulative_hit'],hit)
            state=transition@state; hit+=float(state[goal]); state[goal]=0

    def test_detection_is_monotone_unreachable_and_has_no_faster_than_light_arrival(self):
        for name in ('Two corridors','Blocked exit'):
            maze=Maze(PRESETS[name]); circuit,initial,mapping,goal=walk_operators(maze)
            for frames,_ in (_quantum(circuit,initial,goal,32,True),_classical(initial,mapping,goal,32,True)):
                cumulative=[f['cumulative_hit'] for f in frames]
                self.assertTrue(all(a<=b+1e-14 for a,b in zip(cumulative,cumulative[1:])))
                for frame in frames:
                    self.assertAlmostEqual(sum(frame['probabilities'])+frame['cumulative_hit'],1)
                if name=='Blocked exit': self.assertEqual(cumulative[-1],0)
                else: self.assertTrue(all(abs(v)<1e-14 for v in cumulative[:6]))

    def test_normalization_and_interference(self):
        maze=Maze(PRESETS['Two corridors']); circuit,initial,mapping,goal=walk_operators(maze)
        q,_=_quantum(circuit,initial,goal,32,True); c,_=_classical(initial,mapping,goal,32,True)
        np.testing.assert_allclose(q[1]['probabilities'],c[1]['probabilities'])
        self.assertGreater(max(abs(a-b) for a,b in zip(q[-1]['probabilities'],c[-1]['probabilities'])),.01)
        # Signed amplitudes are necessary; amplitudes cannot be treated as mass.
        self.assertTrue(any(v<0 for frame in q for v in frame['real']))

    def test_no_route_enumeration_and_complete_timing_samples(self):
        with patch('nexqus.maze.check_plan',side_effect=AssertionError('No route oracle allowed')):
            report=compare_walks(Maze(PRESETS['Two corridors']),8)
        self.assertEqual(len(report['quantum']),9); self.assertEqual(report['qubits'],6)
        for key, samples in report['timings']['samples_ms'].items():
            self.assertEqual(len(samples),9)
            self.assertTrue(all(np.isfinite(v) and v>=0 for v in samples))
            self.assertEqual(report['timings']['median_ms'][key],sorted(samples)[4])
        encoded=json.dumps(report,allow_nan=False); self.assertIn('quantum_hit',encoded)
        hostile='</script><img src=x onerror=alert(1)>'
        report['test']=hostile
        self.assertNotIn(hostile,render_walk(report)); self.assertIn('\\u003c/script',render_walk(report))
        for steps in (0,129,True,1.5):
            with self.assertRaises(ValueError): compare_walks(Maze(PRESETS['Two corridors']),steps)


if __name__=='__main__': unittest.main()
