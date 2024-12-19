# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company.
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Utilities for models."""

import math

import torch

from megatron import get_args
from megatron.core.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tensor_model_parallel_group,
)
from megatron.model import RMSNorm, LayerNorm

from deepspeed.runtime.zero import GatheredParameters
from deepspeed.accelerator import get_accelerator

def init_method_normal(sigma):
    """Init method based on N(0, sigma)."""
    def init_(tensor):
        return torch.nn.init.normal_(tensor, mean=0.0, std=sigma)

    return init_


def scaled_init_method_normal(sigma, num_layers):
    """Init method based on N(0, sigma/sqrt(2*num_layers)."""
    std = sigma / math.sqrt(2.0 * num_layers)

    def init_(tensor):
        return torch.nn.init.normal_(tensor, mean=0.0, std=std)

    return init_


def gather_and_init(param, init_method):
    with GatheredParameters(param, modifier_rank=0):
        init_method(param)
        

def perform_masking(attention_scores, attention_mask):
    if attention_mask.dtype == torch.bool:
        attention_scores.masked_fill_(attention_mask, -10000.0)
    else:
        attention_scores.add_(attention_mask)


def attention_mask_func(attention_scores, attention_mask):
    args = get_args()
    if args.curriculum_learning_legacy or args.data_efficiency_curriculum_learning:
        attention_mask_ = attention_mask
        actual_seqlen = attention_scores.size()[2]
        if actual_seqlen != attention_mask_.size()[2]:
            # attention_mask has size [1, 1, seqlen, seqlen]
            attention_mask_ = attention_mask_[:, :, :actual_seqlen, :actual_seqlen].contiguous()
        perform_masking(attention_scores, attention_mask_)
    else:
        perform_masking(attention_scores, attention_mask)
    return attention_scores


def get_linear_layer(rows, columns, init_method, gather_params_on_init=False):
    """Simple linear layer with weight initialization."""
    layer = torch.nn.Linear(rows, columns,
                            device=get_accelerator().current_device_name(),
                            dtype=get_args().params_dtype)
    if get_args().perform_initialization:
        with GatheredParameters(layer.weight, modifier_rank=0, enabled=gather_params_on_init):
            init_method(layer.weight)
    with torch.no_grad():
        with GatheredParameters(layer.bias, modifier_rank=0, enabled=gather_params_on_init):
            layer.bias.zero_()
    return layer

@torch.jit.script
def gelu_impl(x):
    """OpenAI's gelu implementation."""
    return 0.5 * x * (1.0 + torch.tanh(0.7978845608028654 * x *
                                       (1.0 + 0.044715 * x * x)))
def openai_gelu(x):
    return gelu_impl(x)

#This is actually Python equivalent of torch.nn.functional.gelu(), also with type hints for ONNX exporter
@torch.jit.script
def erf_gelu(x):
    return x * 0.5 * (torch.erf(x / 1.41421).to(dtype=x.dtype)+torch.ones_like(x).to(dtype=x.dtype))


@torch.no_grad()
def gather_tensors(input_):
    world_size = get_tensor_model_parallel_world_size()
    rank = get_tensor_model_parallel_rank()

    tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
    tensor_list[rank] = input_.clone()
    torch.distributed.all_gather(tensor_list, input_, group=get_tensor_model_parallel_group())

    return tensor_list


@torch.no_grad()
def compare_tensors(inputs):
    ok = all([x.eq(inputs[0]).all().item() for x in inputs[1:]])
    return ok


def check_equal(inp):
    return compare_tensors(gather_tensors(inp))


def assert_equal(inp, msg=""):
    assert check_equal(inp), msg


def get_check_tp_norm():
    args = get_args()
    check_tp_norm = args.curr_iteration >= args.start_check_tp_norm_iter
    check_tp_norm &= args.curr_iteration <= args.end_check_tp_norm_iter
    check_tp_norm &= args.check_tp_norm
    return check_tp_norm


def tp_norm_module_hook(mod, inp, out, fwd=None, layer_name=""):
    if get_check_tp_norm():
        args = get_args()
        if not isinstance(inp, tuple):
            inputs = [inp]
        else:
            inputs = inp
        if not isinstance(out, tuple):
            outputs = [out]
        else:
            outputs = out

        def get_message(message):
            msg = f"error in {message}, fwd={fwd}"
            if not layer_name:
                return msg
            return msg + f", layer_name = {layer_name}"

        if args.check_tp_norm_type in ["all", "wb"]:
            # compare weight and weight grad
            assert_equal(mod.weight, get_message("mod.weight"))
            assert_equal(mod.weight.grad, get_message("mod.weight.grad"))
            # compare bias and bias grad if present
            if hasattr(mod, "bias"):
                assert_equal(mod.bias, get_message("mod.bias"))
                assert_equal(mod.bias.grad, get_message("mod.bias.grad"))

        if args.check_tp_norm_type in ["all", "io"]:
            # compare inputs
            for i, in_ in enumerate(inputs):
                assert_equal(in_, get_message(f"in_ {i}"))
            # compare outputs
            for i, out_ in enumerate(outputs):
                assert_equal(out_, get_message(f"out_ {i}"))


def layer_name_tp_norm_module_hook(fwd=None, layer_name=""):
    def hook(mod, inp, out):
        tp_norm_module_hook(mod, inp, out, fwd, layer_name)
    return hook


def add_tp_norm_hooks(model, args):
    if args.check_tp_norm:
        for param_name, mod in model.named_modules():
            if isinstance(mod, (RMSNorm, LayerNorm)):
                mod.register_forward_hook(layer_name_tp_norm_module_hook(True, param_name))
                mod.register_full_backward_hook(layer_name_tp_norm_module_hook(False, param_name))
