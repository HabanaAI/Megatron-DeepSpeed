# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company.

from deepspeed.accelerator import get_accelerator

try:
    from apex.normalization import MixedFusedRMSNorm as ApexMixedFusedRMSNorm
except:
    assert False, "Failed: from apex.normalization import MixedFusedRMSNorm"


class RMSNorm(ApexMixedFusedRMSNorm):
    """ Derived class to handle sequence parallel configuration """
    def __init__(self, dim: int, eps: float = 1e-5, **kwargs):
        assert get_accelerator().device_name() == 'cuda', f"Unsupported device: {get_accelerator().device_name()}"
        sequence_parallel = kwargs.pop('sequence_parallel') if 'sequence_parallel' in kwargs else False
        super().__init__(dim, eps, **kwargs)
        if sequence_parallel:
            setattr(self.weight, 'sequence_parallel', sequence_parallel)

    def forward(self, x):
        return super().forward(x)
