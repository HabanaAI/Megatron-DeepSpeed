# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company.

from deepspeed.accelerator import get_accelerator
from megatron import get_args
import torch
from torch.nn import init
from torch.nn.parameter import Parameter

try:
    from habana_frameworks.torch.hpex.normalization import FusedRMSNorm
except:
    FusedRMSNorm = None


# Taken from facebookresearch/llama
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, sequence_parallel=False):
        super().__init__()
        self.eps = eps
        self.weight = Parameter(torch.empty(dim,
                                device=get_accelerator().current_device_name(),
                                dtype=get_args().params_dtype))
        init.ones_(self.weight)
        
        self.use_fused_rmsnorm = get_args().use_fused_rmsnorm

        if sequence_parallel:
            setattr(self.weight, 'sequence_parallel', sequence_parallel)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        if self.use_fused_rmsnorm and x.device.type == "hpu":
            assert FusedRMSNorm is not None, "failed to import FusedRMSNorm"
            return FusedRMSNorm.apply(x, self.weight, self.eps)
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
