"""Strict paired-result comparisons and bounded statistical sanity checks."""
import numpy as np


def compare_results(reference,candidate,precision='double'):
    """Compare complete probabilities and retained states, raising on mismatch.

    Stochastic runs must use paired portable seeds, batch sizes and engines.
    This is a numerical parity check, not a Monte Carlo confidence statement.
    """
    if precision not in ('single','double'): raise ValueError('Unknown precision')
    atol=2e-5 if precision=='single' else 1e-9
    tvd_limit=1e-4 if precision=='single' else 1e-7
    a=np.asarray(reference.probabilities); b=np.asarray(candidate.probabilities)
    if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise AssertionError('Probability shape/nonfinite mismatch')
    for p in (a,b):
        if np.min(p)<-atol or abs(p.sum()-1)>tvd_limit: raise AssertionError('Invalid probabilities')
    if reference.metadata['method']=='trajectory' or candidate.metadata['method']=='trajectory':
        for key in ('method','seed','batch_size','trajectories','random_engine','noise_fastpath'):
            if reference.metadata.get(key)!=candidate.metadata.get(key):
                raise ValueError(f'Paired trajectory mismatch: {key}')
        if reference.metadata.get('random_engine')!='portable':
            raise ValueError('Use portable RNG for cross-backend trajectory parity')
    error=float(np.abs(a-b).max())
    tvd=float(np.abs(a-b).sum()/2)
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
                probability_atol=atol,total_variation_limit=tvd_limit,
                retained_state_max_error=state_error,check='passed')


def trajectory_confidence_bound(trajectories,outcomes,alpha=.01):
    """Hoeffding + union bound for max probability error vs exact channel result.

    Each trajectory contributes a [0,1] probability per outcome. The bound is
    conservative and presumes independent trajectories; it is not backend parity.
    """
    if trajectories<1 or outcomes<1 or not 0<alpha<1: raise ValueError('Invalid bound arguments')
    return float(np.sqrt(np.log(2*outcomes/alpha)/(2*trajectories)))
