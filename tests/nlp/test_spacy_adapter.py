"""Optional real spaCy interoperability checks without any downloaded model."""
import unittest
from spacy.tokens import Doc
from spacy.vocab import Vocab
from nexqus.nlp import DependencyParse


class SpacyAdapterTests(unittest.TestCase):
    def test_real_doc_and_sentence_span_offsets(self):
        doc=Doc(Vocab(),words=['Hi','Alice','likes','Bob'],heads=[0,2,2,2],
                deps=['ROOT','nsubj','ROOT','dobj'])
        parse=DependencyParse.from_spacy(doc[1:])
        self.assertEqual(parse.tokens,('Alice','likes','Bob'))
        self.assertEqual(parse.heads,(1,1,1))
        self.assertEqual(len(parse.circuit([.2,.3,.4]).operations),5)
        with self.assertRaises(ValueError): DependencyParse.from_spacy(doc)
        with self.assertRaises(ValueError): DependencyParse.from_spacy(doc[1:2])

    def test_missing_annotations(self):
        with self.assertRaises(ValueError):
            DependencyParse.from_spacy(Doc(Vocab(),words=['hello']))
