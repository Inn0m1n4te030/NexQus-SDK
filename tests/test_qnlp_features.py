"""Order-aware features and sentence-disjoint sampling; no dataset download."""
from pathlib import Path
import sys
import unittest
import numpy as np
EXAMPLES=Path(__file__).resolve().parents[1]/'examples'
sys.path.insert(0,str(EXAMPLES))
from nlp_similarity import SentenceFeatures
from prepare_mrpc import split_rows,sentences,clean
from train_sentence_pairs import error_groups,fit_model
sys.path.pop(0)


class QNLPFeatureTests(unittest.TestCase):
    def test_bigram_features_preserve_order_before_compression(self):
        fit=['dog chases cat','cat chases dog','dog sleeps quietly','cat runs outside']
        unigram=SentenceFeatures(2,fit_sentences=fit)
        bigram=SentenceFeatures(2,fit_sentences=fit,ngrams=2)
        np.testing.assert_array_equal(unigram.transform(fit[0])['features'],unigram.transform(fit[1])['features'])
        self.assertGreater(np.linalg.norm(np.array(bigram.transform(fit[0])['features'])-bigram.transform(fit[1])['features']),.1)
        before=bigram.projection.copy();bigram.transform('new dog chases cat')
        self.assertNotIn('new',bigram.vocabulary);np.testing.assert_array_equal(before,bigram.projection)
        self.assertLessEqual(len(SentenceFeatures(2,fit_sentences=fit,max_features=3).vocabulary),3)
        with self.assertRaises(ValueError):SentenceFeatures(2,ngrams=3)

    def test_sentence_disjoint_fixed_splits_and_duplicate_conflicts(self):
        train=[dict(left=f'left {i}',right=f'right {i}',label=i%2,category='pair') for i in range(60)]
        test=[dict(left=f'held left {i}',right=f'held right {i}',label=i%2,category='pair') for i in range(30)]
        train.extend(dict(r) for r in test) # deliberately shared source sentences
        a,audit=split_rows(train,test,(20,10,10),17);b,_=split_rows(train,test,(20,10,10),17)
        self.assertEqual(a,b);self.assertTrue(audit['sentence_disjoint'])
        for x,y in [('train','validation'),('train','test'),('validation','test')]:self.assertFalse(sentences(a[x])&sentences(a[y]))
        clean_rows,counts=clean([train[0],dict(train[0],label=1),train[1],train[1]])
        self.assertEqual(len(clean_rows),1);self.assertEqual(counts['conflicting_label_rows_removed'],2)
        self.assertEqual(counts['duplicate_rows_removed'],1)

    def test_groups_are_overlapping_heuristics_and_majority_uses_train_only(self):
        item=dict(left='dog does not run',right='dog runs',category='non_paraphrase',unknown_words=['runs'])
        self.assertEqual(error_groups(item),['non_paraphrase','negation_marker_mismatch','unknown_words'])
        splits=dict(train=[dict(label=1),dict(label=1),dict(label=0)],validation=[dict(label=0)],test=[dict(label=0)])
        result=fit_model('majority',splits,7,1,.05)
        self.assertAlmostEqual(result['predictions']['test'][0],2/3)
        self.assertEqual(result['metrics']['test']['accuracy'],0)
