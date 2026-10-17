"""Regression for real Windows Jupyter stdout fragmentation."""
import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('nexqus_notebook_runner',
    Path(__file__).resolve().parents[1]/'scripts'/'run_notebooks.py')
runner=importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)


def stream(text, name='stdout'):
    return dict(output_type='stream', name=name, text=text)


class NotebookRunnerTests(unittest.TestCase):
    def test_optional_image_skip_fragments_and_separate_gpu_counts(self):
        notebook=dict(cells=[dict(cell_type='code',outputs=[stream('SKIPPED optional '),
            stream(['image GAN: install extra\nSKIPPED optional CUDA: unavailable\n']),
            stream('SKIPPED optional image GAN: not stdout\n','stderr')])])
        self.assertEqual(runner.optional_skip_counts(notebook), {'cuda':1,'image_gan':1})

    def test_windows_split_print(self):
        # Actual Windows artifact: print('NexQus SDK:', version) arrived as
        # two stream outputs, with the version in the second chunk.
        notebook=dict(cells=[dict(cell_type='code', outputs=[
            stream(['NexQus SDK:']), stream([' 1.2.0\n'])])])
        self.assertEqual(runner.reported_sdk_version(notebook), '1.2.0')

    def test_arbitrary_chunks_and_other_output_types(self):
        notebook=dict(cells=[dict(cell_type='code', outputs=[
            stream('Nex'), stream('Qus SDK: '), stream('NexQus SDK: wrong\n','stderr'),
            dict(output_type='display_data', text='NexQus SDK: wrong\n'), stream('1.2.'), stream('0\r\n')])])
        self.assertEqual(runner.reported_sdk_version(notebook), '1.2.0')

    def test_missing_duplicate_and_cell_boundaries(self):
        for cells in ([], [dict(cell_type='markdown', source='NexQus SDK: 1.2.0')],
            [dict(cell_type='code', outputs=[stream('NexQus SDK: 1.2.0\nNexQus SDK: 1.2.0\n')])],
            [dict(cell_type='code', outputs=[stream('NexQus SDK: ')]),
             dict(cell_type='code', outputs=[stream('1.2.0\n')])]):
            self.assertIsNone(runner.reported_sdk_version(dict(cells=cells)))
        self.assertEqual(runner.reported_sdk_version(dict(cells=[dict(cell_type='code',
            outputs=[stream('NexQus SDK: 9.9.9\n')])])), '9.9.9')


if __name__=='__main__': unittest.main()
