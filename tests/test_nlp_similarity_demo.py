"""End-to-end offline NLP example and projector readout ground truths."""
import importlib.util
import json
from pathlib import Path
import threading
import unittest
import urllib.request
import urllib.error
from http.server import ThreadingHTTPServer
import numpy as np
from nexqus import Circuit, Parameter

spec=importlib.util.spec_from_file_location('nlp_similarity',Path(__file__).resolve().parents[1]/'examples/nlp_similarity.py')
demo=importlib.util.module_from_spec(spec);spec.loader.exec_module(demo)


class ProbabilityZeroTests(unittest.TestCase):
    def check_projector(self,**options):
        x=Parameter('x');q=Circuit(2).ry(x,0).ry(2*x,1)
        report=q.explain(bindings={'x':.3},observable='P0',sensitivities=True,**options).to_dict()
        value=np.cos(.15)**2*np.cos(.3)**2
        gradient=-.5*np.sin(.3)*np.cos(.3)**2-np.sin(.6)*np.cos(.15)**2
        self.assertAlmostEqual(report['frames'][-1]['expectation'],value)
        self.assertAlmostEqual(report['sensitivity']['parameters'][0]['derivative'],gradient)
        return report
    def test_projector_analytic_value_and_chain_rule(self):self.check_projector()
    def test_projector_torch_cpu(self):
        try:import torch
        except ImportError:self.skipTest('Optional PyTorch not installed')
        self.check_projector(device='cpu',backend='torch')
    def test_projector_cuda(self):
        try:import torch
        except ImportError:self.skipTest('Optional PyTorch not installed')
        if not torch.cuda.is_available():self.skipTest('CUDA GPU unavailable')
        self.check_projector(device='cuda:0')


class NLPDemoTests(unittest.TestCase):
    def test_scores_independent_of_shared_entangler_and_trace(self):
        engine=demo.SimilarityDemo();data=engine.compare(demo.SAMPLES[0])
        for i,pair in zip([0,2],data['pairs']):
            a,b=data['sentences'][i:i+2]
            expected=float(np.prod(np.cos((np.array(a['angles'])-b['angles'])/2)**2))
            self.assertAlmostEqual(pair['fidelity'],expected)
            raw_a=np.array(a['features']);raw_b=np.array(b['features'])
            self.assertAlmostEqual(pair['classical_cosine'],np.dot(raw_a,raw_b)) # normalized TF-IDF
        self.assertIn('P0',data['report'])
        # Identical sentences have overlap one regardless of truncation.
        identical=engine.compare(demo.SAMPLES[1])
        self.assertAlmostEqual(identical['pairs'][0]['fidelity'],1)
        self.assertAlmostEqual(identical['pairs'][0]['classical_cosine'],1)

    def test_preprocessing_is_frozen_and_oov_is_explicit(self):
        features=demo.SentenceFeatures();minimum=features.minimum.copy();projection=features.projection.copy()
        row=features.transform('quantum unicorn')
        self.assertEqual(row['unknown_words'],['unicorn'])
        features.transform('cat cat cat cat')
        np.testing.assert_array_equal(features.minimum,minimum);np.testing.assert_array_equal(features.projection,projection)
        for sentence in ['', 'unicorn zzz', 'a'*1001, 2]:
            with self.assertRaises(ValueError):features.transform(sentence)
        for q in [True,0,7]:
            with self.assertRaises(ValueError):demo.SentenceFeatures(q)

    def test_external_vectors_and_zero_cosine(self):
        vectors={'first':[1,0,0],'second':[0,1,0],'third':[0,0,1],'fourth':[1,1,1]}
        engine=demo.SimilarityDemo(embeddings=dict(fit_sentences=list(vectors),vectors=vectors))
        data=engine.compare(list(vectors));self.assertAlmostEqual(data['pairs'][0]['classical_cosine'],0)
        with self.assertRaises(ValueError):engine.features.transform('unknown')
        self.assertIsNone(demo.cosine(np.zeros(3),np.ones(3)))
        with self.assertRaises(ValueError):demo.SentenceFeatures(embeddings=dict(fit_sentences=list(vectors),vectors={'first':[float('nan')]}))

    def test_live_server_payload_and_origin_validation(self):
        engine=demo.SimilarityDemo()
        with ThreadingHTTPServer(('127.0.0.1',0),demo.handler(engine,'hello')) as server:
            thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            url=f'http://127.0.0.1:{server.server_port}'
            try:
                self.assertEqual(urllib.request.urlopen(url).read(),b'hello')
                def post(data,origin=None):
                    headers={'Content-Type':'application/json'}
                    if origin:headers['Origin']=origin
                    return urllib.request.urlopen(urllib.request.Request(url+'/api/compare',json.dumps(data).encode(),headers))
                data=json.load(post({'sentences':demo.SAMPLES[0]}));self.assertEqual(len(data['pairs']),2)
                for data in [{'sentences':[]},{'sentences':['zzz']*4},{'extra':1}]:
                    with self.assertRaises(urllib.error.HTTPError) as exc:post(data)
                    self.assertEqual(exc.exception.code,400)
                with self.assertRaises(urllib.error.HTTPError) as exc:post({'sentences':demo.SAMPLES[0]},'https://example.com')
                self.assertEqual(exc.exception.code,403)
            finally:server.shutdown();thread.join()

    def test_export_does_not_allow_markup_injection(self):
        report=demo.render([{'sentences':'</script><script>alert(1)</script>'}])
        self.assertNotIn('</script><script>alert',report)
