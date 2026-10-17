"""Braket SDK integration using only local simulation and mocked AWS."""
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch
import numpy as np
from nexqus import Circuit, Simulator, algorithms, templates
from nexqus.braket import prepare_braket, BraketBackend

# Share fixtures from tests/, without adding src/ to the import path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_braket import ARN, TASK, capabilities, native_result


class BraketIntegrationTests(unittest.TestCase):
    def setUp(self):
        guard=patch('botocore.endpoint.Endpoint.make_request',
                    side_effect=AssertionError('AWS network requests are forbidden in these tests'))
        guard.start(); self.addCleanup(guard.stop)

    def test_local_native_states_match_nexqus(self):
        from braket.devices import LocalSimulator
        cases=[Circuit(1).h(0).x(0).y(0).z(0).s(0).t(0),
               Circuit(4).rx(.5,3).ry(-.7,0).rz(1.4,2).phase(.3,1)
                         .cx(3,0).cz(2,0).cp(.9,1,3).swap(0,2),
               algorithms.qft(4)]
        rng=np.random.default_rng(83)
        q=Circuit(4)
        templates.data_reuploading(q,[.2,-.4,.7,.3],
            rng.normal(size=templates.layer_shape(2,4,rotations='XYZ')),
            rotations='XYZ',entanglement='ring')
        cases.extend([q,q.inverse(),Circuit(8).x(0),Circuit(3)])
        # Block cloud construction throughout genuine local simulator execution.
        with patch('braket.aws.AwsDevice',side_effect=AssertionError('No AWS devices')):
            for q in cases:
                with self.subTest(n=q.n_qubits,gates=q.count_ops()):
                    native=prepare_braket(q).to_braket().state_vector()
                    actual=LocalSimulator().run(native,shots=0).result().values[0]
                    expected=Simulator().run(q,shots=0).statevector
                    np.testing.assert_allclose(actual,expected,atol=1e-11)
            r=prepare_braket(Circuit(8).x(0),shots=17).run_local()
            self.assertEqual(r.counts,{'10000000':17})

    def test_mock_submission_lifecycle_resume_and_failures(self):
        p=prepare_braket(Circuit(2).h(0).cx(0,1),shots=10)
        device=Mock(status='ONLINE'); device.properties.dict.return_value=capabilities()
        task=Mock(); task.id=TASK; task.state.return_value='COMPLETED'
        task.result.return_value=native_result({'00':4,'11':6},[0,1],10)
        task.metadata.return_value={'deviceArn':ARN,'shots':10}
        device.run.return_value=task
        b=BraketBackend(ARN)
        options=dict(allow_paid=True,task_price_usd='.3',shot_price_usd='.001',
                     max_estimated_cost_usd='.32')
        with patch('braket.aws.AwsDevice',return_value=device) as factory:
            job=b.submit(p,**options)
            self.assertEqual(job.id,TASK); self.assertEqual(job.state(),'COMPLETED')
            self.assertEqual(job.result().counts,{'00':4,'11':6})
            job.cancel(); task.cancel.assert_called_once()
            device.run.assert_called_once()
            self.assertEqual(device.run.call_args.kwargs['shots'],10)
            factory.assert_called_once_with(ARN,aws_session=None)
        with patch('braket.aws.AwsQuantumTask',return_value=task):
            resumed=b.get_job(TASK,n_qubits=2,shots=10)
            self.assertEqual(resumed.result().counts,{'00':4,'11':6})
            task.result.return_value=None
            with self.assertRaises(RuntimeError): resumed.result()
            with self.assertRaises(ValueError): b.get_job(TASK,n_qubits=2,shots=11)
        for props,status in [(capabilities(qubits=1),'ONLINE'),(capabilities(),'OFFLINE')]:
            device.reset_mock(); device.properties.dict.return_value=props; device.status=status
            with patch('braket.aws.AwsDevice',return_value=device),self.assertRaises(ValueError):
                b.submit(p,**options)
            device.run.assert_not_called()
        device.reset_mock(); device.status='ONLINE'; device.properties.dict.return_value=capabilities()
        device.run.side_effect=ConnectionError('submission outcome unknown')
        with patch('braket.aws.AwsDevice',return_value=device),self.assertRaises(ConnectionError):
            b.submit(p,**options)
        device.run.assert_called_once()  # never duplicate a potentially accepted paid task


if __name__=='__main__': unittest.main()
