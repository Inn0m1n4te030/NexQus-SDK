"""Inspect optional Torch/CUDA availability: python -m nexqus.devices."""
import json
import os


def device_info():
    info=dict(default_device=os.environ.get('NEXQUS_DEVICE','cpu'),numpy_cpu=True,
              torch_installed=False,cuda_available=False,cuda_devices=[])
    try:
        import torch
    except ImportError: return info
    info.update(torch_installed=True,torch_version=str(torch.__version__),cuda_runtime=torch.version.cuda,
                cuda_available=torch.cuda.is_available())
    if info['cuda_available']:
        for i in range(torch.cuda.device_count()):
            p=torch.cuda.get_device_properties(i)
            info['cuda_devices'].append(dict(device=f'cuda:{i}',name=p.name,memory_bytes=p.total_memory))
    return info


if __name__=='__main__': print(json.dumps(device_info(),indent=2))
