"""Device-local random uniforms with an optional portable counter stream.

Portable stream: a 32-bit integer mixer over a seeded counter, implemented with
int64 arithmetic on both NumPy and Torch. This is a simulation PRNG, not crypto.
It supports fewer than 2**32 draws per run and produces the same uniforms on
CPU/CUDA. Native mode uses the backend's generator and is not cross-backend paired.
"""
import numpy as np


class DeviceRandom:
    def __init__(self,backend,seed=None,engine='portable'):
        if engine not in ('portable','native'): raise ValueError('random_engine must be portable or native')
        self.backend=backend
        self.engine=engine
        self.position=0
        self.key=int(np.random.SeedSequence(seed).generate_state(1)[0])
        self.generator=None
        if engine=='native':
            if backend.torch:
                self.generator=backend.torch.Generator(device=backend.device)
                self.generator.manual_seed(self.key)
            else: self.generator=np.random.default_rng(self.key)

    def uniform(self,count):
        b=self.backend
        if self.engine=='native':
            if b.torch:
                return b.torch.rand(count,device=b.device,dtype=b.torch.float64,generator=self.generator)
            return self.generator.random(count)
        if self.position+count>2**32: raise ValueError('Portable RNG exceeded 2**32 draws; split this run')
        if b.torch:
            x=b.torch.arange(self.position,self.position+count,dtype=b.torch.int64,device=b.device)
        else: x=np.arange(self.position,self.position+count,dtype=np.int64)
        self.position+=count
        x=(x+self.key)&0xffffffff
        x=((x^(x>>16))*0x45d9f3b)&0xffffffff
        x=((x^(x>>16))*0x45d9f3b)&0xffffffff
        x=x^(x>>16)
        if b.torch: x=x.to(b.torch.float64)
        else: x=x.astype(np.float64)
        return (x+.5)/4294967296.0
