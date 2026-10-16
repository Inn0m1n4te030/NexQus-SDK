"""Exact classical-quantum branching and batched sampled dynamic circuits.

Exact branches carry unnormalized density matrices, merging when their entire
classical records coincide. Sampled mode carries one classical record per
trajectory; intermediate quantum/control decisions stay on the selected device.
"""
from collections import Counter
import time
import numpy as np
from .circuit import positive_integer
from .noise import NoiseModel
from .random import DeviceRandom


def run_dynamic(sim,circuit,*,shots,seed,noise,trajectories,initial_state,retain_state,
                readout_error,batch_size,random_engine,max_branches,prepared_arrays,
                compiled,compile_seconds):
    from .simulator import Result, _initial_vector
    if isinstance(shots,bool) or not isinstance(shots,(int,np.integer)) or shots<0:
        raise ValueError('shots must be a nonnegative integer')
    if noise is not None and not isinstance(noise,NoiseModel): raise TypeError('Expected NoiseModel')
    if not np.isfinite(readout_error) or not 0<=readout_error<=1: raise ValueError('Invalid readout_error')
    batch_size=positive_integer(batch_size,'batch_size')
    max_branches=positive_integer(max_branches,'max_branches')
    if sim.method=='statevector':
        raise ValueError('Dynamic circuits need auto/density_matrix for exact branches, or trajectory for sampling')
    method='trajectory' if sim.method=='trajectory' else 'branch_density'
    if method=='trajectory': trajectories=positive_integer(trajectories,'trajectories')
    n=circuit.n_qubits; size=2**n; keys=circuit.classical_keys
    seen=set()
    for op in circuit.operations:
        if op.condition is not None and op.condition[0] not in seen:
            raise ValueError('Classical condition used before its measurement')
        if op.kind=='measure': seen.add(op.data)
    effective_batch=min(batch_size,trajectories) if method=='trajectory' else 1
    b=sim.backend; budget=sim.max_memory_mb*1024**2
    base=b.itemsize*(size if method=='trajectory' else size*size)
    peak_working=base*6*effective_batch
    if peak_working>budget: raise MemoryError('Dynamic simulation exceeds estimated working-memory budget')
    psi=_initial_vector(initial_state,size)
    rng=np.random.default_rng(seed); device_rng=DeviceRandom(b,seed,random_engine)
    cache=dict(prepared_arrays)
    def native(data, permutation=False):
        if id(data) not in cache: cache[id(data)]=sim._prepare_array(data, permutation)
        return cache[id(data)]
    p0=np.diag([1,0]).astype(complex); p1=np.diag([0,1]).astype(complex)
    reset_operators=(p0,np.array([[0,1],[0,0]],complex))
    projectors=(p0,p1)
    def transform(state,data,wires,density=False,diagonal=False,permutation=False):
        prepared = native(data, permutation)
        if density:
            out=b.apply(state,prepared,wires,diagonal,permutation)
            return b.apply(out,prepared if permutation else prepared.conj(),tuple(n+q for q in wires),diagonal,permutation)
        return b.apply_batch(state,prepared,wires,diagonal,permutation)
    def exact_channel(state,operators,q):
        out=b.zeros(state.shape)
        for operator in operators: out+=transform(state,operator,(q,),True)
        return out
    def sampled_channel(state,operators,q):
        count=state.shape[0]
        weights=[]
        for operator in operators:
            v=transform(state,operator,(q,)).reshape(count,-1)
            weights.append((v.conj()*v).real.sum(axis=1))
        weights=b.clamp_min(b.stack(weights,axis=1),0)
        sums=b.clamp_min(weights.sum(axis=1),1e-30)
        cdf=b.cumulative(weights/sums[:,None],1)
        choices=(device_rng.uniform(count)[:,None]>=cdf[:,:-1]).sum(axis=1)
        out=b.zeros(state.shape)
        for index,operator in enumerate(operators):
            v=transform(state,operator,(q,))
            scale=b.where(choices==index,1/b.clamp_min(weights[:,index],1e-30)**.5,0)
            out+=v*scale.reshape((count,)+(1,)*n)
        return out
    def observed(p):
        if not readout_error: return p
        p=p.reshape((2,)*n)
        for q in range(n): p=(1-readout_error)*p+readout_error*np.flip(p,axis=q)
        return p.reshape(-1)
    def label(record):
        return ''.join(str(int(v)) for v in record)
    quantum_counts=Counter(); classical_counts=Counter(); joint_counts=Counter()
    def count_sample(classical,quantum):
        qlabel=format(int(quantum),f'0{n}b')
        quantum_counts[qlabel]+=1
        if keys:
            classical_counts[classical]+=1
            joint_counts[classical+'|'+qlabel]+=1
    b.sync(); started=time.perf_counter(); cpu_started=time.process_time()
    initial=b.array(psi)
    retained=None; classical_probabilities={}; branch_count=None
    if method=='branch_density':
        branches={0:(initial[:,None]*initial.conj()[None,:]).reshape((2,)*(2*n))}
        masks={key:1<<(len(keys)-1-i) for i,key in enumerate(keys)}
        for op in circuit.operations:
            updated={}
            def add(code,state):
                nonlocal peak_working
                new=code not in updated
                estimated=(len(branches)+len(updated)+(1 if new else 0)+4)*base
                peak_working=max(peak_working,estimated)
                if (new and len(updated)>=max_branches) or estimated>budget:
                    raise MemoryError('Exact branch growth exceeds max_branches or estimated memory; use trajectory mode')
                updated[code]=state if new else updated[code]+state
            for code,state in branches.items():
                if op.condition is not None and int(bool(code&masks[op.condition[0]]))!=op.condition[1]:
                    add(code,state); continue
                if op.kind=='measure':
                    mask=masks[op.data]
                    for value,projector in enumerate(projectors):
                        collapsed=transform(state,projector,op.wires,True)
                        weight=float(b.numpy(collapsed.reshape(size,size).diagonal().real.sum()))
                        if weight>0: add((code&~mask)|(mask if value else 0),collapsed)
                elif op.kind=='reset': add(code,exact_channel(state,reset_operators,op.wires[0]))
                elif op.kind=='noise': add(code,exact_channel(state,op.data.kraus,op.wires[0]))
                else:
                    changed=transform(state,op.data,op.wires,True,op.kind=='diagonal',op.kind=='permutation')
                    if noise:
                        for channel in noise.channels_for(op.name):
                            for q in op.wires: changed=exact_channel(changed,channel.kraus,q)
                    add(code,changed)
            branches=updated
        codes=sorted(branches); branch_count=len(codes)
        populations=[]
        total=b.zeros((size,size)) if retain_state else None
        for code in codes:
            state=branches[code].reshape(size,size)
            populations.append(observed(np.array(b.numpy(state.diagonal().real),dtype=float,copy=True)))
            if retain_state: total+=state
        joint=np.stack(populations)
        if not np.isfinite(joint).all() or np.min(joint)<-1e-5 or abs(joint.sum()-1)>1e-3:
            raise RuntimeError('Dynamic density normalization failed')
        joint=np.maximum(joint,0); joint/=joint.sum()
        probabilities=joint.sum(axis=0)
        if keys: classical_probabilities={format(code,f'0{len(keys)}b'):float(joint[i].sum()) for i,code in enumerate(codes)}
        if retain_state: retained=b.numpy(total)
        b.sync(); elapsed=time.perf_counter()-started; cpu_seconds=time.process_time()-cpu_started
        if shots:
            chosen=rng.choice(joint.size,size=shots,p=joint.reshape(-1))
            for value in chosen:
                row,qvalue=divmod(int(value),size)
                count_sample(format(codes[row],f'0{len(keys)}b') if keys else '',qvalue)
    else:
        # Choose which sampled quantum histories supply requested output shots.
        # At each batch end, sample the quantum outcome conditional on that
        # history's classical register, preserving classical/quantum correlation.
        # Storage follows requested output shots, not the total number of histories.
        selected,repeats=np.unique(rng.integers(0,trajectories,size=shots),return_counts=True)
        history_shots=dict(zip(selected.tolist(),repeats.tolist()))
        device_probabilities=b.real_zeros(size); classical_histogram=Counter()
        for offset in range(0,trajectories,effective_batch):
            count=min(effective_batch,trajectories-offset)
            state=b.expand_state(initial,count).reshape((count,)+(2,)*n)
            register={key:b.real_zeros(count) for key in keys}
            for op in circuit.operations:
                condition=None if op.condition is None else register[op.condition[0]]==op.condition[1]
                before=state
                if op.kind=='measure':
                    q=op.wires[0]; tensor=state.reshape(count,2**q,2,2**(n-q-1))
                    a=tensor[:,:,0,:]; d=tensor[:,:,1,:]
                    pop0=(a.conj()*a).real.sum(axis=(1,2)); pop1=(d.conj()*d).real.sum(axis=(1,2))
                    outcome=device_rng.uniform(count)>=pop0/b.clamp_min(pop0+pop1,1e-30)
                    first=b.where(outcome[:,None,None],0,a/b.clamp_min(pop0,1e-30)[:,None,None]**.5)
                    second=b.where(outcome[:,None,None],d/b.clamp_min(pop1,1e-30)[:,None,None]**.5,0)
                    state=b.stack([first,second],axis=2).reshape(state.shape)
                    register[op.data]=outcome
                elif op.kind=='reset': state=sampled_channel(state,reset_operators,op.wires[0])
                elif op.kind=='noise': state=sampled_channel(state,op.data.kraus,op.wires[0])
                else:
                    state=transform(state,op.data,op.wires,diagonal=op.kind=='diagonal',permutation=op.kind=='permutation')
                    if noise:
                        for channel in noise.channels_for(op.name):
                            for q in op.wires: state=sampled_channel(state,channel.kraus,q)
                if condition is not None:
                    state=b.where(condition.reshape((count,)+(1,)*n),state,before)
            population=(state.reshape(count,size).conj()*state.reshape(count,size)).real
            device_probabilities+=population.sum(axis=0)
            records=b.numpy(b.stack([register[key]*1.0 for key in keys],axis=1)) if keys else np.empty((count,0))
            if shots: host_probabilities=b.numpy(population)
            for index,record in enumerate(records):
                classical=label(record); classical_histogram[classical]+=1
                shot_count=history_shots.get(offset+index,0)
                if shot_count:
                    p=observed(np.asarray(host_probabilities[index],float)); p=np.maximum(p,0); p/=p.sum()
                    for qvalue in rng.choice(size,size=shot_count,p=p): count_sample(classical,qvalue)
        probabilities=observed(np.asarray(b.numpy(device_probabilities)/trajectories,float))
        if keys: classical_probabilities={key:value/trajectories for key,value in sorted(classical_histogram.items())}
        b.sync(); elapsed=time.perf_counter()-started; cpu_seconds=time.process_time()-cpu_started
        if not np.isfinite(probabilities).all() or min(probabilities)<-1e-5 or abs(probabilities.sum()-1)>1e-3:
            raise RuntimeError('Dynamic trajectory normalization failed')
        probabilities=np.maximum(probabilities,0); probabilities/=probabilities.sum()
    metadata=dict(n_qubits=n,method=method,backend=b.name,device=b.device,precision=sim.precision,
                  seed=seed,shots=int(shots),trajectories=trajectories if method=='trajectory' else None,
                  batch_size=effective_batch,compiled=compiled,compile_seconds=compile_seconds,
                  simulation_seconds=elapsed,process_cpu_seconds=cpu_seconds,
                  state_storage_bytes=base,estimated_working_bytes=peak_working,
                  gate_cache_bytes=sim.cache_info()['bytes'],operations=len(circuit.operations),
                  readout_error=readout_error,random_engine=random_engine,dynamic=True,
                  classical_keys=list(keys),branch_count=branch_count,max_branches=max_branches,
                  probability_kind='exact' if method=='branch_density' else 'trajectory_estimate',
                  dynamic_noise_kernel='generic_kraus',trajectory_sampling_device=b.device if method=='trajectory' else None)
    if b.torch and b.device.startswith('cuda'): metadata['cuda_allocated_bytes_at_end']=b.torch.cuda.memory_allocated(b.device)
    return Result(probabilities,dict(sorted(quantum_counts.items())),metadata,
                  density_matrix=retained,classical_keys=keys,classical_probabilities=classical_probabilities,
                  classical_counts=dict(sorted(classical_counts.items())),joint_counts=dict(sorted(joint_counts.items())))
