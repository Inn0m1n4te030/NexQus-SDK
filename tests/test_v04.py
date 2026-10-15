"""Analytic gradients checked against finite differences and known functions."""
import unittest
import numpy as np
from nexqus import (Circuit,Parameter,Simulator,Noise,NoiseModel,QuantumFunction,
                    QuantumRegressor,QuantumBinaryClassifier,Adam,SGD)


def finite_difference(function,bindings,name,h=1e-6):
    plus=dict(bindings); minus=dict(bindings)
    plus[name]+=h; minus[name]-=h
    return (function(plus)-function(minus))/(2*h)


class GradientTests(unittest.TestCase):
    def test_single_rotation_known_answer(self):
        t=Parameter('theta'); function=QuantumFunction(Circuit(1).ry(t,0),'Z')
        value,gradient=function.value_and_grad({'theta':.4})
        self.assertAlmostEqual(value,np.cos(.4),places=12)
        self.assertAlmostEqual(gradient['theta'],-np.sin(.4),places=12)

    def test_shared_parameter_gate_local_rule(self):
        t=Parameter('theta'); function=QuantumFunction(Circuit(1).ry(t,0).ry(2*t,0),'Z')
        value,gradient=function.value_and_grad({'theta':.2})
        self.assertAlmostEqual(value,np.cos(.6),places=12)
        self.assertAlmostEqual(gradient['theta'],-3*np.sin(.6),places=12)

    def test_all_gate_rules(self):
        t=Parameter('t')
        for gate in ('rx','ry','rz','phase','cp'):
            q=Circuit(2).h(0).ry(.6,1)
            if gate=='cp': q.cp(t,0,1)
            else: getattr(q,gate)(t,1)
            q.rx(.3,0).h(1).cx(1,0)
            for observable in ('XX','YZ','ZI'):
                function=QuantumFunction(q,observable)
                analytic=function.gradient({'t':.7})['t']
                expected=finite_difference(function,{'t':.7},'t')
                self.assertAlmostEqual(analytic,expected,places=8)

    def test_nonlinear_expression_chain_rule(self):
        a,b=Parameter('a'),Parameter('b')
        q=Circuit(2).ry(a*a+2*b,0).cp(a/(b+2),0,1).rx(1/(a+3),1).h(1)
        function=QuantumFunction(q,'ZX')
        values={'a':.4,'b':.2}
        result=function.gradient(values)
        for name in values: self.assertAlmostEqual(result[name],finite_difference(function,values,name),places=8)

    def test_subset_wrt_inputs_fixed(self):
        x,w,b=Parameter('x'),Parameter('w'),Parameter('b')
        function=QuantumFunction(Circuit(1).ry(w*x+b,0),'Z')
        values={'x':.3,'w':.7,'b':-.2}
        result=function.gradient(values,wrt=[w,b])
        self.assertEqual(set(result),{'w','b'})
        self.assertAlmostEqual(result['w'],-np.sin(.01)*.3,places=12)
        self.assertAlmostEqual(result['b'],-np.sin(.01),places=12)
        self.assertEqual(function.gradient(values,wrt=[]),{})

    def test_inverse_and_composition_gradients(self):
        t=Parameter('t'); block=Circuit(2).ry(t,0).cp(2*t,0,1)
        q=Circuit(2).h(1).compose(block.inverse()).rx(.3,1)
        function=QuantumFunction(q,'XZ')
        self.assertAlmostEqual(function.gradient({'t':.5})['t'],finite_difference(function,{'t':.5},'t'),places=8)

    def test_noisy_exact_gradient(self):
        t=Parameter('t')
        function=QuantumFunction(Circuit(1).ry(t,0),'Z',noise=NoiseModel(Noise.amplitude_damping(.2)))
        value,g=function.value_and_grad({'t':.4})
        self.assertAlmostEqual(value,.2+.8*np.cos(.4),places=12)
        self.assertAlmostEqual(g['t'],-.8*np.sin(.4),places=12)

    def test_noise_gate_names_preserved(self):
        t=Parameter('t'); q=Circuit(1).ry(t,0).inverse()
        noise=NoiseModel().after('RY†',Noise.amplitude_damping(.25))
        function=QuantumFunction(q,'Z',noise=noise)
        self.assertAlmostEqual(function.gradient({'t':.3})['t'],-.75*np.sin(.3),places=12)

    def test_constant_function(self):
        function=QuantumFunction(Circuit(1).h(0),'X')
        value,gradient=function.value_and_grad({})
        self.assertAlmostEqual(value,1.,places=12)
        self.assertEqual(gradient,{})

    def test_validation_errors(self):
        t=Parameter('t'); q=Circuit(1).ry(t,0)
        with self.assertRaises(ValueError): QuantumFunction(q,'XX')
        with self.assertRaises(ValueError): QuantumFunction(q,'A')
        with self.assertRaises(ValueError): QuantumFunction(q,'Z',Simulator(method='trajectory'))
        function=QuantumFunction(q,'Z')
        for wrt in (['typo'],['t','t'],[2]):
            with self.assertRaises(ValueError): function.gradient({'t':.2},wrt)
        with self.assertRaises(ValueError): function.gradient({})
        with self.assertRaises(ValueError): function.gradient({'t':float('nan')})


class OptimizerTests(unittest.TestCase):
    def test_sgd_one_step(self):
        parameters={'x':1.}; result=SGD(lr=.1).step(parameters,{'x':2.})
        self.assertEqual(parameters,{'x':1.})
        self.assertAlmostEqual(result['x'],.8)

    def test_adam_first_step_and_reset(self):
        opt=Adam(lr=.1)
        result=opt.step({'x':1.},{'x':2.})
        self.assertAlmostEqual(result['x'],.9,places=8)
        self.assertEqual(opt.t,1)
        opt.reset(); self.assertEqual(opt.t,0)
        self.assertEqual(opt.m,{})

    def test_adam_quadratic(self):
        params={'x':2.}; opt=Adam(lr=.1)
        for _ in range(80): params=opt.step(params,{'x':2*params['x']})
        self.assertLess(abs(params['x']),.05)

    def test_optimizer_errors(self):
        for opt in (SGD(),Adam()):
            with self.assertRaises(ValueError): opt.step({'x':1},{'typo':1})
            with self.assertRaises(ValueError): opt.step({'x':1},{'x':np.inf})
        with self.assertRaises(ValueError): SGD(lr=0)
        with self.assertRaises(ValueError): Adam(beta1=1)
        opt=Adam(); opt.step({'x':1},{'x':1})
        with self.assertRaises(ValueError): opt.step({'y':1},{'y':1})


class LearningTests(unittest.TestCase):
    def function(self):
        x,w,b=Parameter('x'),Parameter('w'),Parameter('b')
        return QuantumFunction(Circuit(1).ry(w*x+b,0),'Z')

    def test_sine_learning(self):
        X=np.linspace(-np.pi,np.pi,12); y=np.sin(X)
        model=QuantumRegressor(self.function(),['x'],{'w':.7,'b':-.8})
        before=float(np.mean((model.predict(X)-y)**2))
        self.assertIs(model.fit(X,y,epochs=60,optimizer=Adam(lr=.05)),model)
        test_x=np.linspace(-np.pi,np.pi,61)
        after=float(np.mean((model.predict(test_x)-np.sin(test_x))**2))
        self.assertLess(after,.002)
        self.assertLess(after,before/20)
        self.assertEqual(len(model.history),60)

    def test_binary_classifier(self):
        X=np.array([-1.2,-.7,-.3,.3,.7,1.2]); y=(X>0).astype(int)
        model=QuantumBinaryClassifier(self.function(),['x'],{'w':.7,'b':-1.0})
        model.fit(X,y,epochs=35,optimizer=Adam(lr=.06))
        np.testing.assert_array_equal(model.predict(X),y)
        scores=model.predict_proba(X)
        np.testing.assert_allclose(scores.sum(axis=1),1)
        self.assertTrue(np.all((scores>=0)&(scores<=1)))

    def test_multiple_input_columns(self):
        x,y,w=Parameter('x'),Parameter('y'),Parameter('w')
        function=QuantumFunction(Circuit(1).ry(w*x+y,0),'Z')
        model=QuantumRegressor(function,['x','y'],{'w':.5})
        X=np.array([[.2,.1],[.4,.3]])
        np.testing.assert_allclose(model.predict(X),np.cos(.5*X[:,0]+X[:,1]))

    def test_training_validation(self):
        f=self.function()
        with self.assertRaises(ValueError): QuantumRegressor(f,['x'],{'w':.5})
        with self.assertRaises(ValueError): QuantumRegressor(f,['x'],{'x':.5,'w':.5,'b':0})
        model=QuantumRegressor(f,['x'],{'w':.5,'b':0})
        with self.assertRaises(ValueError): model.predict([[1,2]])
        with self.assertRaises(ValueError): model.fit([1,2],[1])
        with self.assertRaises(ValueError): model.fit([1,2],[2,3])
        with self.assertRaises(ValueError): model.fit([1,2],[0,1],epochs=0)
        classifier=QuantumBinaryClassifier(f,['x'],{'w':.5,'b':0})
        with self.assertRaises(ValueError): classifier.fit([1,2],[0,2])


try:
    import torch
except ImportError:
    torch=None


@unittest.skipIf(torch is None,'Optional PyTorch not installed')
class TorchV04Tests(unittest.TestCase):
    def compare(self,device):
        t=Parameter('t'); q=Circuit(2).ry(t,0).h(1).cp(2*t,0,1).rz(t*t,1)
        values={'t':.4}
        for noise in (None,NoiseModel(Noise.amplitude_damping(.17))):
            ref=QuantumFunction(q,'XX',noise=noise)
            function=QuantumFunction(q,'XX',Simulator(backend='torch',device=device),noise)
            va,ga=ref.value_and_grad(values); vb,gb=function.value_and_grad(values)
            self.assertAlmostEqual(va,vb,places=10)
            self.assertAlmostEqual(ga['t'],gb['t'],places=10)

    def test_torch_cpu_v04(self): self.compare('cpu')

    @unittest.skipUnless(torch is not None and torch.cuda.is_available(),'CUDA GPU unavailable')
    def test_torch_cuda_v04(self): self.compare('cuda:0')


if __name__=='__main__': unittest.main()
