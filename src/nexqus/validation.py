"""Strict paired-result comparisons and bounded statistical sanity checks."""
import numpy as np
from .circuit import positive_integer


def _validated_result(result, atol, tvd_limit):
    """Reject corrupt data before comparing differences (NaN > tol is false)."""
    n=positive_integer(result.metadata['n_qubits'],'n_qubits')
    size=2**n
    p=np.asarray(result.probabilities)
    if p.shape!=(size,) or not np.isfinite(p).all():
        raise AssertionError('Probability shape/nonfinite mismatch')
    if np.iscomplexobj(p) or np.min(p)<-atol or abs(p.sum()-1)>tvd_limit:
        raise AssertionError('Invalid probabilities')
    if result.statevector is not None and result.density_matrix is not None:
        raise AssertionError('A result cannot retain both state representations')
    for name,shape in [('statevector',(size,)),('density_matrix',(size,size))]:
        value=getattr(result,name)
        if value is None: continue
        value=np.asarray(value)
        if value.shape!=shape or not np.isfinite(value).all():
            raise AssertionError('Retained state shape/nonfinite mismatch')
        if name=='statevector':
            if abs(np.vdot(value,value)-1)>atol:
                raise AssertionError('Retained statevector is not normalized')
        else:
            if abs(np.trace(value)-1)>atol or np.max(abs(value-value.conj().T))>atol:
                raise AssertionError('Retained density must be Hermitian with unit trace')
    classical=result.classical_probabilities
    if result.classical_keys:
        if not classical: raise AssertionError('Missing classical probabilities')
        for key,value in classical.items():
            if (not isinstance(key,str) or len(key)!=len(result.classical_keys)
                    or set(key)-set('01') or not np.isscalar(value)
                    or not np.isrealobj(value) or not np.isfinite(value)
                    or value < -atol or value > 1+atol):
                raise AssertionError('Invalid classical probabilities')
        if abs(sum(classical.values())-1)>tvd_limit:
            raise AssertionError('Classical probabilities are not normalized')
    elif classical:
        raise AssertionError('Classical probabilities require classical keys')
    return p


def compare_results(reference,candidate,precision='double'):
    """Compare complete probabilities and retained states, raising on mismatch.

    Stochastic runs must use paired portable seeds, batch sizes and engines.
    This is a numerical parity check, not a Monte Carlo confidence statement.
    """
    if precision not in ('single','double'): raise ValueError('Unknown precision')
    atol=2e-5 if precision=='single' else 1e-9
    tvd_limit=1e-4 if precision=='single' else 1e-7
    a=_validated_result(reference,atol,tvd_limit)
    b=_validated_result(candidate,atol,tvd_limit)
    if a.shape!=b.shape:
        raise AssertionError('Probability shape/nonfinite mismatch')
    for name in ('statevector','density_matrix'):
        if (getattr(reference,name) is None)!=(getattr(candidate,name) is None):
            raise AssertionError('Retained state representation/presence mismatch')
    if reference.metadata['method']=='trajectory' or candidate.metadata['method']=='trajectory':
        if reference.metadata.get('seed') is None or candidate.metadata.get('seed') is None:
            raise ValueError('Paired trajectory comparison requires an explicit seed')
        for key in ('method','seed','batch_size','trajectories','random_engine','noise_fastpath'):
            if reference.metadata.get(key)!=candidate.metadata.get(key):
                raise ValueError(f'Paired trajectory mismatch: {key}')
        if reference.metadata.get('random_engine')!='portable':
            raise ValueError('Use portable RNG for cross-backend trajectory parity')
    error=float(np.abs(a-b).max())
    tvd=float(np.abs(a-b).sum()/2)
    if reference.classical_keys!=candidate.classical_keys: raise AssertionError('Classical key order mismatch')
    classical_keys=set(reference.classical_probabilities)|set(candidate.classical_probabilities)
    differences=[abs(reference.classical_probabilities.get(k,0)-candidate.classical_probabilities.get(k,0))
                 for k in classical_keys]
    classical_error=max(differences,default=0.)
    if classical_error>atol or sum(differences)/2>tvd_limit:
        raise AssertionError('Classical probability mismatch')
    if error>atol or tvd>tvd_limit:
        raise AssertionError(f'Probability mismatch: max={error}, TVD={tvd}')
    state_error=None
    if reference.statevector is not None and candidate.statevector is not None:
        x=reference.statevector; y=candidate.statevector
        overlap=np.vdot(x,y)
        phase=overlap/abs(overlap) if abs(overlap)>0 else 1
        state_error=float(np.max(abs(x-y/phase)))
    elif reference.density_matrix is not None and candidate.density_matrix is not None:
        state_error=float(np.max(abs(reference.density_matrix-candidate.density_matrix)))
    if state_error is not None and state_error>atol: raise AssertionError('Retained state mismatch')
    return dict(max_probability_error=error,total_variation_distance=tvd,
                max_classical_probability_error=float(classical_error),
                probability_atol=atol,total_variation_limit=tvd_limit,
                retained_state_max_error=state_error,check='passed')


def trajectory_confidence_bound(trajectories,outcomes,alpha=.01):
    """Hoeffding + union bound for max probability error vs exact channel result.

    Each trajectory contributes a [0,1] probability per outcome. The bound is
    conservative and presumes independent trajectories; it is not backend parity.
    """
    trajectories=positive_integer(trajectories,'trajectories')
    outcomes=positive_integer(outcomes,'outcomes')
    if not np.isscalar(alpha) or not np.isrealobj(alpha) or not np.isfinite(alpha) or not 0<alpha<1:
        raise ValueError('Invalid bound arguments')
    return float(np.sqrt(np.log(2*outcomes/alpha)/(2*trajectories)))
