"""Bounded, exact local pure-state traces and evidence-based explanations."""
import copy
import html
import json
from pathlib import Path
import numbers

import numpy as np

from .backend import Backend
from .circuit import Circuit, I, X, Y, Z, H, positive_integer
from .drawing import _label
from .parameters import bindings_dict, real_scalar
from .simulator import Simulator, Result


def _action(op):
    """Recognize actual matrices, not names supplied by a caller."""
    if op.kind == 'permutation':
        return 'Reorders the selected basis amplitudes according to the stored permutation.'
    matrix = op.data
    if op.kind == 'diagonal':
        if len(op.wires) > 2:
            return 'Applies the stored basis-state phases. Basis probabilities stay unchanged at this step.'
        matrix = np.diag(matrix)
    fixed = [(I, 'Identity leaves the state unchanged.'),
             (H, 'Hadamard mixes the zero and one amplitudes. Its effect depends on the incoming state.'),
             (X, 'X exchanges the zero and one amplitudes on this wire.'),
             (Y, 'Y exchanges the zero and one amplitudes and changes their phases.'),
             (Z, 'Z changes the sign of the one amplitude. Later mixing gates can reveal this phase change.')]
    cx = np.eye(4); cx[2:,2:] = X.real
    fixed.extend([(cx, 'Controlled X flips the second listed wire when the first is one. It can create entanglement.'),
                  (np.diag([1,1,1,-1]), 'Controlled Z changes the sign of the amplitude where both listed wires are one.'),
                  (np.eye(4)[[0,2,1,3]], 'SWAP exchanges the quantum states associated with the two listed wires.')])
    for candidate, text in fixed:
        if matrix.shape == candidate.shape and np.allclose(matrix,candidate,atol=1e-12,rtol=0):
            return text
    if len(op.params) == 1 and isinstance(op.params[0],numbers.Real):
        for angle in (float(op.params[0]), -float(op.params[0])):
            if matrix.shape == (2,2):
                for axis, pauli in [('X',X),('Y',Y),('Z',Z)]:
                    candidate=np.cos(angle/2)*I-1j*np.sin(angle/2)*pauli
                    if np.allclose(matrix,candidate,atol=1e-12,rtol=0):
                        return f'Rotates this qubit by {angle:.6g} radians around its Bloch {axis} axis.'
    if op.kind == 'diagonal':
        return 'Applies the stored basis-state phases. Basis probabilities stay unchanged at this step.'
    return 'Applies the stored unitary on the selected wires. It can change amplitudes and phases while preserving normalization.'


def _blocks(blocks, count):
    checked=[]
    for block in blocks or ():
        if not isinstance(block,dict): raise ValueError('Each block must be a dictionary')
        name=block.get('name'); start=block.get('start'); stop=block.get('stop')
        role=block.get('role','block')
        if (not isinstance(name,str) or not name.strip() or
                any(isinstance(v,bool) or not isinstance(v,numbers.Integral) for v in (start,stop))
                or not 0 <= start < stop <= count):
            raise ValueError('Block needs a name and a valid half-open operation range start:stop')
        if role not in ('encoding','trainable','entangling','readout','block'):
            raise ValueError('Unknown block role')
        checked.append(dict(name=name,start=int(start),stop=int(stop),role=role))
    checked.sort(key=lambda b:b['start'])
    if any(a['stop']>b['start'] for a,b in zip(checked,checked[1:])):
        raise ValueError('Explanation blocks must not overlap')
    return checked


def _snapshot(psi, n, signs, observable, previous):
    probabilities=abs(psi)**2
    if not np.isfinite(psi).all() or not np.isclose(probabilities.sum(),1,atol=1e-10,rtol=0):
        raise ValueError('Trace encountered an invalid pure quantum state')
    result=Result(probabilities,{},dict(n_qubits=n),statevector=psi)
    qubits=[]
    for wire in range(n):
        rho=result.reduced_density([wire])
        eigenvalues=np.clip(np.linalg.eigvalsh(rho),0,1)
        entropy=float(-sum(x*np.log2(x) for x in eigenvalues if x>1e-15))
        qubits.append(dict(wire=wire,bloch=[float(2*rho[0,1].real),float(-2*rho[0,1].imag),
                                         float((rho[0,0]-rho[1,1]).real)],
            purity=float(np.clip(np.sum(abs(rho)**2),.5,1)),entropy=max(0,entropy)))
    means=signs@probabilities
    covariance=(signs*probabilities)@signs.T-np.outer(means,means)
    nonzero=np.flatnonzero(probabilities>1e-14)
    reference=int(nonzero[0])
    phases=np.angle(psi*np.exp(-1j*np.angle(psi[reference])))
    phase_change=None
    if previous is not None:
        common=np.flatnonzero((abs(previous)**2>1e-14)&(probabilities>1e-14))
        if len(common)>1:
            differences=np.angle(psi[common]*previous[common].conj())
            phase_change=float(np.max(abs(np.angle(np.exp(1j*(differences-differences[0]))))))
    return dict(probabilities=probabilities.tolist(),amplitudes=np.column_stack((psi.real,psi.imag)).tolist(),
        phases=[float(phases[i]) if probabilities[i]>1e-14 else None for i in range(len(psi))],
        phase_reference=format(reference,f'0{n}b'),qubits=qubits,z_covariance=covariance.tolist(),
        expectation=result.expectation(observable),
        probability_change=None if previous is None else float(np.sum(abs(probabilities-abs(previous)**2))/2),
        phase_change=phase_change,
        fidelity_to_previous=None if previous is None else float(np.clip(abs(np.vdot(previous,psi))**2,0,1)))


class HTMLReport(str):
    """Self-contained HTML; notebook display uses an isolated script-enabled frame."""
    def _repr_html_(self):
        return ('<iframe title="NexQus circuit explanation" sandbox="allow-scripts allow-downloads" '
                'style="width:100%;height:980px;border:0" srcdoc="'+html.escape(self,quote=True)+'"></iframe>')


class Explanation:
    """A calculated trace. Export contains states/parameter values supplied by you."""
    def __init__(self, data):
        self._data=data

    def to_dict(self):
        return copy.deepcopy(self._data)

    def to_html(self, *, filename=None):
        from .explain_html import render_report
        report=HTMLReport(render_report(self._data))
        if filename is not None: Path(filename).write_text(report,encoding='utf-8')
        return report

    def _repr_html_(self):
        return self.to_html()._repr_html_()


def explain(circuit, *, bindings=None, observable=None, blocks=None, parameter_roles=None,
            initial_state=None, title='Inside the circuit', max_qubits=10, max_steps=256,
            max_report_mb=32,backend=None,device=None, interventions=None, sensitivities=False, comparisons=None, training_history=None):
    """Trace operations on CPU/CUDA; export snapshot diagnostics to host JSON.

    No cloud access, sampling, external JS or language model. Rejects noise and
    dynamic circuits. Blocks are caller annotations using half-open op ranges;
    parameter_roles labels parameters as feature/weight/parameter, not attribution.
    interventions is up to 16 dictionaries with a zero-based operation index and
    either skip=True or angle=<radians>. Optional axis is rx/ry/rz; it is
    required for ambiguous matrices (for example a zero-angle rotation). Angle edits require an actual RX/RY/RZ
    matrix with a recorded numerical angle. Alternatives are independent edits
    of the bound circuit, calculated in Python and embedded for offline selection.
    A block edit uses block=<index in start-sorted annotations>, skip=True.
    sensitivities=True computes signed local derivatives of the final expectation
    for up to 32 symbolic bindings (including shared/expression parameters).
    Feature roles refer to encoded inputs, not raw text or preprocessed features.
    comparisons contains up to 8 named complete binding dictionaries.
    training_history contains up to 32 ordered epoch/bindings/optional loss records.
    Additional traces use the same circuit, observable, initial state and backend.
    Comparisons inherit interventions; checkpoints omit them. Neither nests series.
    Limits are checked before state/trace allocation and again after serialization.
    """
    if not isinstance(sensitivities,bool): raise ValueError('sensitivities must be a boolean')
    if not isinstance(circuit,Circuit): raise TypeError('Expected a Circuit')
    if sensitivities and len(circuit.parameters)>32:
        raise ValueError('Sensitivity reports support at most 32 symbolic parameters')
    max_qubits=positive_integer(max_qubits,'max_qubits')
    max_steps=positive_integer(max_steps,'max_steps')
    limit=real_scalar(max_report_mb)
    if limit<=0: raise ValueError('max_report_mb must be positive')
    n=circuit.n_qubits; count=len(circuit.operations)
    if n>max_qubits or count>max_steps: raise ValueError('Circuit exceeds explainability max_qubits/max_steps')
    # Conservative JSON/state workspace estimate; avoids allocating oversized state histories.
    if (count+1)*((2**n)*160+n*n*32+n*256)>limit*1024**2:
        raise ValueError('Estimated trace exceeds max_report_mb; reduce circuit or explicitly raise the limit')
    if not isinstance(title,str) or not title.strip(): raise ValueError('title must be a nonempty string')
    if circuit.dynamic or any(op.kind=='noise' for op in circuit.operations):
        raise ValueError('Explainability currently supports static pure-state circuits; noise/dynamic operations are unsupported')
    if observable is None: observable='Z'+'I'*(n-1)
    if not isinstance(observable,str) or len(observable)!=n or set(observable)-set('IXYZ'):
        raise ValueError('observable must be a Pauli string in q0-first order')
    values=bindings_dict({} if bindings is None else bindings)
    cases=[] if comparisons is None else list(comparisons)
    history=[] if training_history is None else list(training_history)
    if len(cases)>8 or len(history)>32:
        raise ValueError('At most 8 comparisons and 32 training checkpoints per report')
    for case in cases:
        if (not isinstance(case,dict) or set(case)!={'name','bindings'}
                or not isinstance(case['name'],str) or not case['name'].strip()):
            raise ValueError('Each comparison needs exactly a nonempty name and complete bindings')
        circuit.bind(bindings_dict(case['bindings']))
    previous_epoch=-1
    for record in history:
        if (not isinstance(record,dict) or set(record)-{'epoch','bindings','loss'}
                or not {'epoch','bindings'}<=set(record)):
            raise ValueError('Each training checkpoint needs epoch, complete bindings and optional loss')
        epoch=record['epoch']
        if isinstance(epoch,bool) or not isinstance(epoch,numbers.Integral) or epoch<=previous_epoch or epoch>2**53-1:
            raise ValueError('Checkpoint epochs must be nonnegative strictly increasing safe integers')
        previous_epoch=epoch
        circuit.bind(bindings_dict(record['bindings']))
        if 'loss' in record and real_scalar(record['loss'])<0:
            raise ValueError('Checkpoint loss must be nonnegative')
    estimated=(count+1)*((2**n)*160+n*n*32+n*256)
    if estimated*(1+len(cases)+len(history))>limit*1024**2:
        raise ValueError('Estimated comparison/checkpoint traces exceed max_report_mb')

    annotations=_blocks(blocks,count)
    roles={} if parameter_roles is None else dict(parameter_roles)
    if set(roles)-set(circuit.parameters) or any(v not in ('feature','weight','parameter') for v in roles.values()):
        raise ValueError('parameter_roles must label known parameters as feature/weight/parameter')
    # Binding validates all parameter names and produces a new operation list.
    bound=circuit.bind(values)
    operations=[]
    for index,(original,op) in enumerate(zip(tuple(circuit.operations),bound.operations)):
        if op.kind not in ('unitary','diagonal','permutation'):
            raise ValueError(f'Unsupported explanation operation: {op.kind}')
        parameters=sorted(original.data[1].names) if original.kind=='parameterized' else []
        block=next((i for i,b in enumerate(annotations) if b['start']<=index<b['stop']),None)
        operations.append(dict(label=_label(op,6),wires=list(op.wires),kind=op.kind,
                               action=_action(op),parameters=parameters,block=block))
    sim=Simulator.for_device(device,backend=backend,max_memory_mb=max(1,limit))
    backend=sim.backend
    psi=sim.run(Circuit(n),shots=0,initial_state=initial_state,native_result=True).statevector
    indices=np.arange(2**n)
    signs=np.array([1-2*((indices>>(n-1-q))&1) for q in range(n)])
    frames=[_snapshot(backend.numpy(psi),n,signs,observable,None)]
    frames[0]['explanation']='Initial supplied pure state.' if initial_state is not None else 'Every qubit starts in zero.'
    for op,info in zip(bound.operations,operations):
        previous=psi
        data=backend.index_array(np.argsort(op.data)) if op.kind=='permutation' else backend.array(op.data)
        psi=backend.apply(psi.reshape((2,)*n),data,op.wires,
                          diagonal=op.kind=='diagonal',permutation=op.kind=='permutation').reshape(-1)
        frame=_snapshot(backend.numpy(psi),n,signs,observable,backend.numpy(previous))
        tv=frame['probability_change']
        evidence=f' Basis-probability total variation changed by {tv:.6g}.'
        if tv<1e-12 and frame['phase_change'] is not None and frame['phase_change']>1e-10:
            evidence+=' Relative phases changed despite unchanged basis probabilities.'
        entangled=[f'q{q["wire"]}' for q in frame['qubits'] if q['purity']<1-1e-10]
        if entangled:
            evidence+=' Reduced states of '+', '.join(entangled)+' are mixed; each is entangled with the remaining qubits in this pure global state.'
        frame['explanation']=info['action']+evidence
        frames.append(frame)
    from . import __version__
    data=dict(schema_version=1,sdk_version=__version__,title=title,n_qubits=n,
        mode='exact local pure-state simulation',observable=observable,
        execution=dict(backend=backend.name,device=backend.device,precision='double',
                       snapshot_diagnostics='CPU NumPy; explicit per-frame HTML/JSON export'),
        depth=bound.depth(),gate_counts=bound.count_ops(gates_only=True),
        operations=operations,blocks=annotations,frames=frames,
        parameters=[dict(name=k,value=values[k],role=roles.get(k,'parameter')) for k in sorted(values)],
        notes=['Snapshots are simulated states, not hidden states observed on a QPU.',
               'Bloch vectors describe reduced single-qubit states; they do not describe the full joint state.',
               'Z covariance measures basis-dependent correlation; it is not an entanglement test.',
               'Reduced entropy quantifies qubit-versus-rest entanglement here because the global state is pure.',
               'Block/parameter roles are caller annotations. Local derivatives and circuit edits are not global feature importance or real-world causal effects.'])
    if sensitivities:
        from .gradients import QuantumFunction

        class InitialStateFunction(QuantumFunction):
            def _evaluate(self, candidate):
                return self.simulator.run(candidate, shots=0, initial_state=initial_state,
                                          native_result=True).expectation(self.observable)

        gradients=InitialStateFunction(circuit,observable,sim).gradient(values)
        data['sensitivity']=dict(
            scope='Original circuit final Pauli expectation at the supplied bindings',
            method='Gate-local analytic parameter shift with arithmetic chain rule',
            output=frames[-1]['expectation'],
            parameters=[dict(name=k,role=roles.get(k,'parameter'),value=values[k],
                             derivative=float(gradients[k])) for k in sorted(gradients)],
            interpretation='Signed local derivative per binding unit; feature derivatives refer to encoded inputs. '
                           'Magnitudes depend on parameter units and scaling. A zero derivative can be a stationary point, '
                           'not irrelevance. No classifier transform, global attribution or causal importance is inferred.')
    requests=[] if interventions is None else list(interventions)
    if len(requests)>16: raise ValueError('At most 16 interventions per report')
    alternatives=[]
    for request in requests:
        if not isinstance(request,dict): raise ValueError('Each intervention must be a dictionary')
        if set(request)=={'block','skip'} and request['skip'] is True:
            block_index=request['block']
            if (isinstance(block_index,bool) or not isinstance(block_index,numbers.Integral)
                    or not 0<=block_index<len(annotations)):
                raise ValueError('Intervention block must index the sorted explanation blocks')
            block=annotations[block_index]
            edited=copy.deepcopy(bound)
            for index in range(block['start'],block['stop']):
                wires=bound.operations[index].wires
                edited.operations[index]=Circuit(n).diagonal(np.ones(2**len(wires)),wires).operations[0]
            variant=explain(edited,observable=observable,blocks=annotations,initial_state=initial_state,
                title=title,max_qubits=max_qubits,max_steps=max_steps,max_report_mb=limit,
                backend=backend.name,device=backend.device).to_dict()
            for a,b in zip(variant['operations'],operations): a['parameters']=b['parameters']
            variant['parameters']=copy.deepcopy(data['parameters'])
            variant['intervention']=dict(request)
            alternatives.append(dict(label=f'Skip block {block_index+1}: {block["name"]}',trace=variant))
            if len(json.dumps(dict(data,alternatives=alternatives),ensure_ascii=False,allow_nan=False).encode('utf-8'))>limit*1024**2:
                raise ValueError('Combined intervention traces exceed max_report_mb')
            continue
        index=request.get('operation')
        if isinstance(index,bool) or not isinstance(index,numbers.Integral) or not 0<=index<count:
            raise ValueError('Intervention operation must be a valid zero-based index')
        if set(request)=={'operation','skip'} and request['skip'] is True:
            # Identity retains the stage indices and caller block boundaries.
            edited=copy.deepcopy(bound)
            replacement=Circuit(n).diagonal(np.ones(2**len(bound.operations[index].wires)),bound.operations[index].wires)
            edited.operations[index]=replacement.operations[0]
            description=f'Skip operation {index+1}: {operations[index]["label"]}'
        elif set(request) in ({'operation','angle'}, {'operation','angle','axis'}):
            angle=real_scalar(request['angle']); op=bound.operations[index]
            if op.kind!='unitary' or len(op.wires)!=1 or len(op.params)!=1:
                raise ValueError('Angle edits require a recorded RX, RY or RZ rotation')
            axes=[]
            for name,pauli in [('rx',X),('ry',Y),('rz',Z)]:
                original=float(op.params[0])
                if np.allclose(op.data,np.cos(original/2)*I-1j*np.sin(original/2)*pauli,atol=1e-12,rtol=0):
                    axes.append(name)
            if 'axis' in request:
                axes=[a for a in axes if a==request['axis']]
            if len(axes)!=1: raise ValueError('Rotation axis is ambiguous or unsupported; supply axis=rx, ry or rz matching the original matrix')
            axis=axes[0]
            edited=copy.deepcopy(bound)
            edited.operations[index]=getattr(Circuit(n),axis)(angle,op.wires[0]).operations[0]
            description=f'Operation {index+1}: {axis.upper()}({angle:.6g})'
        else: raise ValueError('Intervention needs exactly operation and skip=True or angle')
        variant=explain(edited,observable=observable,blocks=annotations,initial_state=initial_state,
            title=title,max_qubits=max_qubits,max_steps=max_steps,max_report_mb=limit,
            backend=backend.name,device=backend.device).to_dict()
        # Keep the original symbolic occurrence labels as context, not a new binding.
        for a,b in zip(variant['operations'],operations): a['parameters']=b['parameters']
        variant['parameters']=copy.deepcopy(data['parameters'])
        variant['intervention']=dict(request)
        alternatives.append(dict(label=description,trace=variant))
        # Enforce the aggregate budget after every alternative, not just each trace.
        if len(json.dumps(dict(data,alternatives=alternatives),ensure_ascii=False,allow_nan=False).encode('utf-8'))>limit*1024**2:
            raise ValueError('Combined intervention traces exceed max_report_mb')
    data['alternatives']=alternatives
    def extra_trace(bindings, include_interventions=False):
        return explain(circuit,bindings=bindings,observable=observable,blocks=annotations,
            parameter_roles=roles,initial_state=initial_state,title=title,max_qubits=max_qubits,
            max_steps=max_steps,max_report_mb=limit,backend=backend.name,device=backend.device,
            sensitivities=sensitivities,
            interventions=requests if include_interventions else None).to_dict()

    def check_series_size():
        if len(json.dumps(data,ensure_ascii=False,allow_nan=False).encode('utf-8'))>limit*1024**2:
            raise ValueError('Combined comparison/checkpoint traces exceed max_report_mb')

    if cases:
        data['comparisons']=[]
        for case in cases:
            data['comparisons'].append(dict(name=case['name'],trace=extra_trace(case['bindings'],True)))
            check_series_size()
    if history:
        data['training_history']=[]
        for record in history:
            checkpoint=dict(epoch=int(record['epoch']),trace=extra_trace(record['bindings']))
            if 'loss' in record: checkpoint['loss']=real_scalar(record['loss'])
            data['training_history'].append(checkpoint)
            check_series_size()

    if len(json.dumps(data,ensure_ascii=False,allow_nan=False).encode('utf-8'))>limit*1024**2:
        raise ValueError('Serialized trace exceeds max_report_mb')
    return Explanation(data)
