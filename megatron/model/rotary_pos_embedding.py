# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company.
# coding=utf-8

# The following code has been taken from https://github.com/NVIDIA/NeMo/blob/ \
# 782b4e1652aaa43c8be390d9db0dc89544afa080/nemo/collections/nlp/modules/ \
# common/megatron/rotary_pos_embedding.py

import importlib.util
from megatron.global_vars import get_args
import torch

from torch import einsum, nn

__all__ = ['RotaryEmbedding', 'apply_rotary_pos_emb']

try:
    from habana_frameworks.torch.hpex.kernels import RotaryPosEmbeddingHelperV1
except ImportError:
    RotaryPosEmbeddingHelperV1 = None


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, theta=10000):
        super().__init__()
        args = get_args()
        self.eval_hf_rope = args.eval_hf_rope
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        if self.eval_hf_rope and inv_freq.dtype != args.params_dtype:
            inv_freq = inv_freq.to(args.params_dtype)
        self.register_buffer('inv_freq', inv_freq)
        if importlib.util.find_spec('einops') is None:
            raise RuntimeError("einops is required for Rotary Embedding")

    def forward(self, max_seq_len, offset=0):
        if self.eval_hf_rope and self.inv_freq.dtype != torch.float32:
            self.inv_freq = self.inv_freq.float()
        seq = torch.arange(max_seq_len, device=self.inv_freq.device) + offset
        freqs = einsum('i , j -> i j', seq.type_as(self.inv_freq), self.inv_freq)
        # first part even vector components, second part odd vector components,
        #  2 * dim in dimension size
        emb = torch.cat((freqs, freqs), dim=-1)
        # emb [seq_length, .., dim]
        from einops import rearrange
        return rearrange(emb, 'n d -> n 1 1 d')


def _rotate_half(x):
    """
    change sign so the last dimension becomes [-odd, +even]
    """
    from einops import rearrange
    x = rearrange(x, '... (j d) -> ... j d', j=2)
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(t, freqs, cos_cached, sin_cached):
    """
    input tensor t is of shape [seq_length, ..., dim]
    rotary positional embedding tensor freqs is of shape [seq_length, ..., dim]
    check https://kexue.fm/archives/8265 for detailed formulas
    """
    rot_dim = freqs.shape[-1]
    t_pass = None
    # due to 0 dim of t_pass tensor, there is zeros tensor DMA from H2D which
    # affects performance, check whether we need t_pass
    if t.shape[-1] != rot_dim:
        # ideally t_pass is empty so rotary pos embedding is applied to all tensor t
        t, t_pass = t[..., :rot_dim], t[..., rot_dim:]

    if cos_cached is None or sin_cached is None or t.shape[0] != cos_cached.shape[0]:
        freqs_ = freqs[:t.shape[0]]
        cos_cached = freqs_.cos().to(t.dtype)
        sin_cached = freqs_.sin().to(t.dtype)

    if t.device.type == "hpu":
        assert RotaryPosEmbeddingHelperV1 is not None, "failed to import RotaryPosEmbeddingHelperV1"
        t = RotaryPosEmbeddingHelperV1.apply(t, cos_cached, sin_cached, 0) # offset already used in RotaryEmbedding.forward
    else:
        # first part is cosine component
        # second part is sine component, need to change signs with _rotate_half method
        t = (t * cos_cached) + (_rotate_half(t) * sin_cached)
    if t_pass is None:
        return t
    return torch.cat((t, t_pass), dim=-1)
