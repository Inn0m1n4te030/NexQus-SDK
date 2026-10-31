"""Training fixture protocol, gradient and encoder checks."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
from nexqus import QuantumDistribution
EXAMPLES=Path(__file__).resolve().parents[1]/'examples'
sys.path.insert(0,str(EXAMPLES))
import train_sentence_pairs as experiment
from nlp_similarity import SentenceFeatures
sys.path.pop(0)


class SentenceTrainingTests(unittest.TestCase):
    def test_trainable_shared_weights_have_correct_nonzero_gradients(self):
        q=experiment.pair_template();model=QuantumDistribution(q)
        bindings=dict(a0=.2,a1=.7,b0=1.3,b1=.4,w0=.6,w1=-.4,w2=.3,w3=.2)
        _,gradient=model.value_and_grad(bindings,wrt=['w0','w1','w2','w3'])
        self.assertGreater(max(abs(v[0]) for v in gradient.values()),1e-3)
        for key,g in gradient.items():
            plus=dict(bindings);minus=dict(bindings);plus[key]+=1e-6;minus[key]-=1e-6
            np.testing.assert_allclose(g,(model(plus)-model(minus))/2e-6,atol=1e-8)

    def test_classical_gradients_independent_finite_difference(self):
        rng=np.random.default_rng(3);x=rng.normal(size=4)
        for kind,size in [('linear',5),('mlp',49)]:
            params={f'p{i}':float(v) for i,v in enumerate(rng.normal(0,.2,size))}
            _,gradient=experiment.classical_value_grad(x,params,kind)
            for key,g in gradient.items():
                a=dict(params);b=dict(params);a[key]+=1e-6;b[key]-=1e-6
                expected=(experiment.classical_value_grad(x,a,kind)[0]-experiment.classical_value_grad(x,b,kind)[0])/2e-6
                self.assertAlmostEqual(g,expected,places=8)

    def test_split_validation_and_frozen_training_fit(self):
        rows=experiment.load_pairs(EXAMPLES/'data/sentence_pairs.csv')
        self.assertEqual([sum(r['split']==s for r in rows) for s in ['train','validation','test']],[18,8,12])
        training=list(dict.fromkeys(r[k] for r in rows if r['split']=='train' for k in ['left','right']))
        features=SentenceFeatures(2,fit_sentences=training)
        self.assertNotIn('quietly',features.vocabulary)
        a=features.transform('the student chases the dog');b=features.transform('the dog chases the student')
        np.testing.assert_array_equal(a['angles'],b['angles'])
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'bad.csv';source=(EXAMPLES/'data/sentence_pairs.csv').read_text()
            path.write_text(source+'test,the cat sleeps,the cat rests,1,duplicate\n')
            with self.assertRaisesRegex(ValueError,'Duplicate'):experiment.load_pairs(path)

    def test_checkpoint_selection_and_reproducibility(self):
        item=dict(label=1,bindings=dict(a0=.2,a1=.7,b0=1.3,b1=.4),classical=np.array([.1,.2,.3,.4]))
        splits={s:[item] for s in ['train','validation','test']}
        a=experiment.fit_model('quantum',splits,7,2,.05)
        b=experiment.fit_model('quantum',splits,7,2,.05)
        self.assertEqual(a['selected_epoch'],min(a['history'],key=lambda r:r['validation_loss'])['epoch'])
        self.assertEqual(a['predictions'],b['predictions'])
        self.assertLess(a['history'][-1]['train_loss'],a['history'][0]['train_loss'])
