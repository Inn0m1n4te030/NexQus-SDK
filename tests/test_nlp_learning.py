import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from nexqus import SentencePairClassifier

class SentencePairLearningTests(unittest.TestCase):
    def test_fit_gradient_explanation_and_checkpoint(self):
        x=np.array([[[0.,.2],[.1,.3]],[[0.,0.],[2.,2.]],[[.4,.5],[.5,.6]],[[.2,.1],[2.5,2.]]])
        y=[1,0,1,0]
        m=SentencePairClassifier().fit(x,y,epochs=5)
        self.assertLess(m.history_[-1]['loss'],m.history_[0]['loss'])
        np.testing.assert_allclose(m.predict_proba(x).sum(axis=1),1)
        np.testing.assert_allclose(m.predict_proba(x),m.predict_proba(x[:,::-1]),atol=1e-12)
        d=m.explain(x[0],comparison=x[1]).to_dict()
        self.assertAlmostEqual(d['frames'][-1]['expectation'],m.predict_proba(x[:1])[0,1])
        h=1e-5; changed=x[:1].copy();changed[0,0,0]+=h
        plus=m.predict_proba(changed)[0,1];changed[0,0,0]-=2*h
        derivative=(plus-m.predict_proba(changed)[0,1])/(2*h)
        a=next(v for v in d['sensitivity']['parameters'] if v['name']=='a0')
        self.assertAlmostEqual(a['derivative'],derivative,places=7)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'model.json';m.save(p);r=SentencePairClassifier.load(p)
            np.testing.assert_allclose(r.predict_proba(x),m.predict_proba(x))
            data=json.loads(p.read_text());data['parameters']['w0']=float('nan');p.write_text(json.dumps(data))
            with self.assertRaises(ValueError):SentencePairClassifier.load(p)

    def test_validation_selects_best_and_bad_inputs(self):
        x=[[[0.],[.3]],[[0.],[2.]]]; m=SentencePairClassifier(1)
        with self.assertRaises(ValueError):m.predict(x)
        m.fit(x,[1,0],epochs=3,validation_data=(x,[0,1]))
        self.assertEqual(m.selected_epoch_,min(m.history_,key=lambda r:r['validation_loss'])['epoch'])
        for bad in ([],[[1,2]],[[[float('nan')],[0]]]):
            with self.assertRaises(ValueError):m.predict(bad)
        with self.assertRaises(ValueError):m.fit(x,[2,0])
