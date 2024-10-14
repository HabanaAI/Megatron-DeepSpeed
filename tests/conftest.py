# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company.
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import os
import pytest

from deepspeed.accelerator import get_accelerator

# from megatron import initialize_megatron
from megatron.core import parallel_state
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

from megatron.core.transformer.transformer_config import TransformerConfig

import torch

world_size = int(os.getenv("WORLD_SIZE", '1'))
rank = int(os.getenv('RANK', '0'))
torch.distributed.init_process_group(
    backend=get_accelerator().communication_backend_name(),
    world_size=world_size, rank=rank)

tp_world_size = world_size
pp_world_size = 1
assert world_size == (tp_world_size * pp_world_size)

# initialize model parallel for tests
parallel_state.set_tensor_model_parallel_world_size(tp_world_size)
parallel_state.set_tensor_model_parallel_rank(rank)
# parallel_state._set_global_memory_buffer()
parallel_state.set_pipeline_model_parallel_world_size(pp_world_size)
parallel_state.set_pipeline_model_parallel_rank(rank)
parallel_state.initialize_model_parallel()

model_parallel_cuda_manual_seed(123)


@pytest.fixture
def transformer_config():
    print(f"transformer_config")
    return TransformerConfig(num_layers=2, hidden_size=12, num_attention_heads=4, use_cpu_initialization=True)
