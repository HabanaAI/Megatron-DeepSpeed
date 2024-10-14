# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company.
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.tensor_parallel import layers, random
from megatron.core.transformer import TransformerConfig
from megatron.global_vars import set_args
from .commons import set_random_seed
from .commons import print_separator
from .commons import initialize_distributed
from megatron.core import mpu
from megatron.model.transformer import ParallelAttention, ParallelTransformerLayer
import os
import pytest
from torch.nn.parameter import Parameter
import torch.nn.init as init
import torch
import types
from deepspeed.accelerator import get_accelerator
import sys
sys.path.append("../..")


@pytest.fixture
def tensor_model_parallel_size():
    return int(os.getenv("WORLD_SIZE", '1'))


device_name = get_accelerator().device_name()
def test_parallel_embedding(tensor_model_parallel_size):

    initialize_distributed()
    if torch.distributed.get_rank() == 0:
        print('> testing parallel embedding with model parallel size {} ...'.
              format(tensor_model_parallel_size))

    mpu.initialize_model_parallel(tensor_model_parallel_size)
    tensor_model_parallel_size = mpu.get_tensor_model_parallel_world_size()

    batch_size = 17
    seq_length = 23
    vocab_size = 48
    hidden_size = 16
    seed = 1236

    set_random_seed(123)
    input_data = torch.LongTensor(
        size=(batch_size, seq_length)).random_(0, vocab_size).to(device_name)
    loss_weight = torch.randn([batch_size, seq_length, hidden_size]).to(device_name)

    set_random_seed(seed)
    embedding_original = torch.nn.Embedding(vocab_size, hidden_size).to(device_name)

    output = embedding_original(input_data)
    loss_original = torch.mul(output, loss_weight).sum()
    loss_original.backward()

    set_random_seed(seed)
    args = types.SimpleNamespace(embed_layernorm=False)
    set_args(args)
    config = ModelParallelConfig(tensor_model_parallel_size, perform_initialization=True, use_cpu_initialization=True)
    embedding_vocab_parallel = layers.VocabParallelEmbedding(
        vocab_size, hidden_size, init_method=init.normal_, config=config).to(device_name)
    output = embedding_vocab_parallel(input_data)
    loss_vocab_parallel = torch.mul(output, loss_weight).sum()
    loss_vocab_parallel.backward()

    torch.distributed.barrier()
    error = loss_vocab_parallel.sub(loss_original).abs()
    print('   error in loss (vocab parallel) on global rank {}: {}'.format(
        torch.distributed.get_rank(), error))
    assert error < 1.0e-12, 'error: {}'.format(error)

    weight_grad_orig = torch.split(embedding_original.weight.grad,
                                   vocab_size // tensor_model_parallel_size,
                                   0)[mpu.get_tensor_model_parallel_rank()]
    error = embedding_vocab_parallel.weight.grad.sub(
        weight_grad_orig).abs().max()
    print('   error in grad (vocab parallel) on global rank {}: {}'.format(
        torch.distributed.get_rank(), error))
    assert error < 1.0e-6, 'error: {}'.format(error)

    # Reset groups
    mpu.destroy_model_parallel()

    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        print('>> passed the test :-)')


def test_initialize_affine_weight(tensor_model_parallel_size):

    initialize_distributed()
    def delete_if_exist(tensor):
        if hasattr(tensor, 'tensor_model_parallel'):
            delattr(tensor, 'tensor_model_parallel')
        if hasattr(tensor, 'partition_dim'):
            delattr(tensor, 'partition_dim')
        if hasattr(tensor, 'partition_stride'):
            delattr(tensor, 'partition_stride')
    mpu.initialize_model_parallel(tensor_model_parallel_size)
    if torch.distributed.get_rank() == 0:
        print('> testing initialize_affine_weight with model parallel '
              'size: {}'.format(tensor_model_parallel_size))
    tensor_model_parallel_size = mpu.get_tensor_model_parallel_world_size()

    seed = 12345
    input_size_coeff = 1
    input_size = input_size_coeff * tensor_model_parallel_size
    output_size_coeff = 1
    output_size = output_size_coeff * tensor_model_parallel_size

    # ---------------
    # Column parallel
    # ---------------
    weight = torch.empty(output_size_coeff, input_size)
    set_random_seed(seed)
    for _ in range(mpu.get_tensor_model_parallel_rank() + 1):
        delete_if_exist(weight)
        layers._initialize_affine_weight_gpu(weight, torch.nn.init.normal_, 0)
    # Target.
    set_random_seed(seed)
    master_weight = torch.empty(output_size, input_size)
    torch.nn.init.normal_(master_weight)
    rank = mpu.get_tensor_model_parallel_rank()
    my_weight = torch.split(master_weight, output_size_coeff,
                            dim=0)[rank].contiguous().clone()

    # Compare.
    error = weight.sub(my_weight).abs().max()
    torch.distributed.barrier()
    print('   column parallel max error (should be zero) on global rank '
          '{}: {}'.format(torch.distributed.get_rank(), error))
    assert error < 1.0e-6

    # ------------
    # Row parallel
    # ------------
    weight = torch.empty(output_size, input_size_coeff)
    set_random_seed(seed)
    for _ in range(mpu.get_tensor_model_parallel_rank() + 1):
        delete_if_exist(weight)
        layers._initialize_affine_weight_gpu(weight, torch.nn.init.normal_, 1)
    # Target.
    set_random_seed(seed)
    master_weight = torch.empty(output_size, input_size)
    torch.nn.init.normal_(master_weight)
    master_weight_list = torch.split(master_weight.reshape(-1), output_size*input_size_coeff)
    master_weight_list = [master_weight_list_.reshape(output_size, input_size_coeff) for master_weight_list_ in master_weight_list]
    master_weight = torch.cat(master_weight_list, dim=1)
    rank = mpu.get_tensor_model_parallel_rank()
    my_weight = torch.split(master_weight, input_size_coeff,
                            dim=1)[rank].contiguous().clone()

    # Compare.
    error = weight.sub(my_weight).abs().max()
    torch.distributed.barrier()
    print('   row parallel max error (should be zero) on global rank '
          '{}: {}'.format(torch.distributed.get_rank(), error))
    assert error < 1.0e-6

    # Reset groups
    mpu.destroy_model_parallel()

    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        print(' >> passed the test :-)')


class IdentityLayer2D(torch.nn.Module):
    def __init__(self, m, n):
        super(IdentityLayer2D, self).__init__()
        self.weight = Parameter(torch.Tensor(m, n))
        torch.nn.init.xavier_normal_(self.weight)

    def forward(self):
        return self.weight


def test_column_parallel_linear(tensor_model_parallel_size):

    initialize_distributed()
    mpu.initialize_model_parallel(tensor_model_parallel_size)
    if torch.distributed.get_rank() == 0:
        print('> testing ColumnParallelLinear with model parallel '
              'size: {}'.format(tensor_model_parallel_size))
    tensor_model_parallel_size = mpu.get_tensor_model_parallel_world_size()

    seed = 12345
    set_random_seed(seed)
    input_size_coeff = 13
    input_size = input_size_coeff * tensor_model_parallel_size
    output_size_coeff = 17
    output_size = output_size_coeff * tensor_model_parallel_size
    batch_size = 7

    # Network
    identity_layer = IdentityLayer2D(batch_size, input_size).to(device_name)
    args = types.SimpleNamespace(transformer_impl='local')
    set_args(args)
    config = ModelParallelConfig(tensor_model_parallel_size, perform_initialization=True, use_cpu_initialization=True)
    init_method = torch.nn.init.xavier_normal_
    linear_layer = layers.ColumnParallelLinear(
        input_size, output_size, config=config, init_method=init_method, keep_master_weight_for_test=True, gather_output=True).to(device_name)
    loss_weight = torch.randn([batch_size, output_size]).to(device_name)
    # Forward
    input_ = identity_layer()
    output = linear_layer(input_)[0] # skip bias output
    loss = torch.mul(output, loss_weight).sum()
    # Backward
    loss.backward()

    # Values.
    dLdY = loss_weight
    X = identity_layer.weight
    A = linear_layer.master_weight.to(device_name)
    dLdA = torch.matmul(dLdY.t(), X)
    dLdb = torch.matmul(torch.ones(batch_size, 1).to(device_name).t(), dLdY).view(-1)
    dLdX = torch.matmul(dLdY, A)

    rank = mpu.get_tensor_model_parallel_rank()
    my_dLdA = torch.split(dLdA, output_size_coeff,
                          dim=0)[rank].contiguous().clone()
    error = my_dLdA.sub(linear_layer.weight.grad).abs().max()
    torch.distributed.barrier()
    print('   error in dLdA on global rank {}: {}'.format(
        torch.distributed.get_rank(), error))
    assert error < 1.0e-6

    my_dLdb = torch.split(dLdb, output_size_coeff,
                          dim=0)[rank].contiguous().clone()
    error = my_dLdb.sub(linear_layer.bias.grad).abs().max()
    torch.distributed.barrier()
    print('   error in dLdb on global rank {}: {}'.format(
        torch.distributed.get_rank(), error))
    assert error < 1.0e-6

    error = dLdX.sub(identity_layer.weight.grad).abs().max()
    torch.distributed.barrier()
    print('   error in dLdX on global rank {}: {}'.format(
        torch.distributed.get_rank(), error))
    assert error < 1.0e-6

    # Reset groups
    mpu.destroy_model_parallel()

    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        print(' >> passed the test :-)')


def test_row_parallel_linear(tensor_model_parallel_size):

    initialize_distributed()
    mpu.initialize_model_parallel(tensor_model_parallel_size)
    if torch.distributed.get_rank() == 0:
        print('> testing RowParallelLinear with model parallel '
              'size: {}'.format(tensor_model_parallel_size))
    tensor_model_parallel_size = mpu.get_tensor_model_parallel_world_size()

    seed = 12345
    set_random_seed(seed)
    input_size_coeff = 13
    input_size = input_size_coeff * tensor_model_parallel_size
    output_size_coeff = 17
    output_size = output_size_coeff * tensor_model_parallel_size
    batch_size = 7

    # Network
    identity_layer = IdentityLayer2D(batch_size, input_size).to(device_name)
    args = types.SimpleNamespace(transformer_impl='local')
    set_args(args)
    config = ModelParallelConfig(tensor_model_parallel_size, perform_initialization=True, use_cpu_initialization=True)
    init_method = torch.nn.init.xavier_normal_
    linear_layer = layers.RowParallelLinear(
        input_size, output_size, config=config, init_method=init_method, keep_master_weight_for_test=True).to(device_name)
    loss_weight = torch.randn([batch_size, output_size]).to(device_name)
    # Forward
    input_ = identity_layer()
    output = linear_layer(input_)[0] # skip bias output
    loss = torch.mul(output, loss_weight).sum()
    # Backward
    loss.backward()

    # Values.
    dLdY = loss_weight
    X = identity_layer.weight
    A = linear_layer.master_weight.to(device_name)
    dLdA = torch.matmul(dLdY.t(), X)
    dLdb = torch.matmul(torch.ones(batch_size, 1).to(device_name).t(), dLdY).view(-1)
    dLdX = torch.matmul(dLdY, A)

    rank = mpu.get_tensor_model_parallel_rank()
    my_dLdA = torch.split(dLdA, input_size_coeff,
                          dim=1)[rank].contiguous().clone()
    error = my_dLdA.sub(linear_layer.weight.grad).abs().max()
    torch.distributed.barrier()
    print('   error in dLdA on global rank {}: {}'.format(
        torch.distributed.get_rank(), error))
    assert error < 1.0e-6

    error = dLdb.sub(linear_layer.bias.grad).abs().max()
    torch.distributed.barrier()
    print('   error in dLdb on global rank {}: {}'.format(
        torch.distributed.get_rank(), error))
    assert error < 1.0e-6

    error = dLdX.sub(identity_layer.weight.grad).abs().max()
    torch.distributed.barrier()
    print('   error in dLdX on global rank {}: {}'.format(
        torch.distributed.get_rank(), error))
    assert error < 1.0e-6

    # Reset groups
    mpu.destroy_model_parallel()

    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        print(' >> passed the test :-)')


class IdentityLayer3D(torch.nn.Module):
    def __init__(self, m, n, k):
        super(IdentityLayer3D, self).__init__()
        self.weight = Parameter(torch.Tensor(m, n, k))
        torch.nn.init.xavier_normal_(self.weight)

    def forward(self):
        return self.weight


def parallel_self_attention(tensor_model_parallel_size, num_att_heads_per_partition,
                            hidden_size_per_att_head, dropout_prob, batch_size,
                            sequence_length):
    mpu.initialize_model_parallel(tensor_model_parallel_size)
    tensor_model_parallel_size = mpu.get_tensor_model_parallel_world_size()

    seed = 12345
    set_random_seed(seed)

    num_att_heads = num_att_heads_per_partition * \
        torch.distributed.get_world_size()
    hidden_size = hidden_size_per_att_head * num_att_heads

    # Network
    identity_layer = IdentityLayer3D(batch_size, sequence_length,
                                     hidden_size).to(device_name)
    args = types.SimpleNamespace(use_flash_attn_v1=False,
                                 use_flash_attn_triton=False,
                                 use_flash_attn_v2=False,
                                 use_flash_attn_builder=False,
                                 add_bias_linear=False,
                                 force_ds_sequence_parallel=False,
                                 num_attention_heads=num_att_heads,
                                 use_alibi_position_embeddings=False,
                                 seq_length=sequence_length,
                                 micro_batch_size=batch_size,
                                 transformer_impl="local",
                                 use_fast_softmax="None")
    set_args(args)
    config = TransformerConfig(tensor_model_parallel_size,
                               hidden_size=hidden_size,
                               num_attention_heads=num_att_heads,
                               attention_dropout=dropout_prob,
                               num_layers=1,
                               perform_initialization=True,
                               use_cpu_initialization=True)
    attention_layer = ParallelAttention(config, layer_number=1).to(device_name)
    loss_weight = torch.randn([batch_size, sequence_length, hidden_size]).to(device_name)
    attention_mask = torch.randn([batch_size, 1, 1, sequence_length]).to(device_name)
    # Forward
    input_ = identity_layer()
    output = attention_layer(input_, attention_mask)[0] # skip bias output
    loss = torch.mul(output, loss_weight).sum()
    # Backward
    loss.backward()

    rank = mpu.get_tensor_model_parallel_rank()
    mpu.destroy_model_parallel()
    return rank, hidden_size, tensor_model_parallel_size, loss, \
        attention_layer, identity_layer


def test_parallel_self_attention(tensor_model_parallel_size):

    initialize_distributed()
    if torch.distributed.get_rank() == 0:
        print('> testing ParallelSelfAttention with model parallel '
              'size: {}'.format(tensor_model_parallel_size))

    num_att_heads_per_partition = 3
    hidden_size_per_att_head = 7
    dropout_prob = 0.0  # has to be zero
    batch_size = 5
    sequence_length = 13

    rank_1, hideen_size_1, tensor_model_parallel_size_1, loss_1, \
        attention_layer_1, identity_layer_1 = parallel_self_attention(
            1, num_att_heads_per_partition,
            hidden_size_per_att_head, dropout_prob, batch_size, sequence_length)

    rank, hidden_size, tensor_model_parallel_size, loss, \
        attention_layer, identity_layer = parallel_self_attention(
            tensor_model_parallel_size, num_att_heads_per_partition,
            hidden_size_per_att_head, dropout_prob, batch_size, sequence_length)
    assert hideen_size_1 == hidden_size

    error = loss_1.sub(loss).abs().max()
    torch.distributed.barrier()
    print('   loss error on global rank {}: {}'.format(
        torch.distributed.get_rank(), error))
    assert error < 5.0e-6

    my_lin_grad = torch.split(
        attention_layer_1.query_key_value.weight.grad,
        attention_layer.query_key_value.weight.grad.shape[0], 0)[rank]
    error = my_lin_grad.sub(
        attention_layer.query_key_value.weight.grad).abs().max()
    torch.distributed.barrier()
    print('   weight gradient error on global rank {}: {}'.format(
        torch.distributed.get_rank(), error))
    assert error < 5.0e-6

    error = identity_layer_1.weight.grad.sub(
        identity_layer.weight.grad).abs().max()
    torch.distributed.barrier()
    print('   input gradient error on global rank {}: {}'.format(
        torch.distributed.get_rank(), error))
    assert error < 5.0e-6

    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        print(' >> passed the test :-)')


def parallel_transformer(tensor_model_parallel_size, num_att_heads_per_partition,
                         hidden_size_per_att_head, batch_size, sequence_length):

    mpu.initialize_model_parallel(tensor_model_parallel_size)
    tensor_model_parallel_size = mpu.get_tensor_model_parallel_world_size()

    seed = 12345
    set_random_seed(seed)

    num_att_heads = num_att_heads_per_partition * \
        torch.distributed.get_world_size()
    hidden_size = hidden_size_per_att_head * num_att_heads
    intermediate_size = 4 * hidden_size

    # Network
    identity_layer = IdentityLayer3D(batch_size, sequence_length,
                                     hidden_size).to(device_name)
    args = types.SimpleNamespace(use_flash_attn_v1=False,
                                 use_flash_attn_triton=False,
                                 use_flash_attn_v2=False,
                                 use_flash_attn_builder=False,
                                 add_bias_linear=False,
                                 force_ds_sequence_parallel=False,
                                 num_attention_heads=num_att_heads,
                                 use_alibi_position_embeddings=False,
                                 seq_length=sequence_length,
                                 micro_batch_size=batch_size,
                                 transformer_impl="local",
                                 use_fast_softmax="None",
                                 normalization='layernorm',
                                 no_persist_layer_norm=False,
                                 apply_layernorm_1p=False,
                                 mem_efficient_ln=False,
                                 params_dtype=torch.float32,
                                 num_experts_switch=None,
                                 swiglu=False,
                                 openai_gelu=False,
                                 onnx_safe=False,
                                 squared_relu=False,
                                 bias_gelu_fusion=False,
                                 retro_add_retriever=False)
    set_args(args)
    config = TransformerConfig(tensor_model_parallel_size,
                               hidden_size=hidden_size,
                               num_attention_heads=num_att_heads,
                               ffn_hidden_size=intermediate_size,
                               attention_dropout=0.0,
                               hidden_dropout=0.0,
                               num_layers=1,
                               perform_initialization=True,
                               use_cpu_initialization=True)
    transformer_layer = ParallelTransformerLayer(config, layer_number=1).to(device_name)

    loss_weight = torch.randn([batch_size, sequence_length, hidden_size]).to(device_name)
    attention_mask = torch.randn([batch_size, 1, 1, sequence_length]).to(device_name)
    # Forward
    input_ = identity_layer()
    output = transformer_layer(input_, attention_mask)[0] # skip bias output
    loss = torch.mul(output, loss_weight).sum()
    # Backward
    loss.backward()

    rank = mpu.get_tensor_model_parallel_rank()
    mpu.destroy_model_parallel()
    return rank, hidden_size, tensor_model_parallel_size, loss, \
        transformer_layer, identity_layer


def test_parallel_transformer_layer(tensor_model_parallel_size):

    initialize_distributed()
    if torch.distributed.get_rank() == 0:
        print('> testing ParallelTransformerLayer with model parallel '
              'size: {}'.format(tensor_model_parallel_size))

    num_att_heads_per_partition = 3
    hidden_size_per_att_head = 7
    batch_size = 5
    sequence_length = 13

    rank_1, hidden_size_1, tensor_model_parallel_size_1, loss_1, \
        transformer_layer_1, identity_layer_1 = parallel_transformer(
            1, num_att_heads_per_partition,
            hidden_size_per_att_head, batch_size, sequence_length)

    rank, hidden_size, tensor_model_parallel_size, loss, \
        transformer_layer, identity_layer = parallel_transformer(
            tensor_model_parallel_size, num_att_heads_per_partition,
            hidden_size_per_att_head, batch_size, sequence_length)

    error = loss_1.sub(loss).abs().max()
    torch.distributed.barrier()
    print('   loss error on global rank {}: {}'.format(
        torch.distributed.get_rank(), error))
    assert error < 5.0e-5, 'error: {}'.format(error)

    error = identity_layer_1.weight.grad.sub(
        identity_layer.weight.grad).abs().max()
    torch.distributed.barrier()
    print('   input gradient error on global rank {}: {}'.format(
        torch.distributed.get_rank(), error))
    assert error < 5.0e-5, 'error: {}'.format(error)

    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        print(' >> passed the test :-)')


if __name__ == '__main__':

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    initialize_distributed()
    world_size = torch.distributed.get_world_size()

    print_separator('test initialize affine weight')
    tensor_model_parallel_size = 1
    while tensor_model_parallel_size <= world_size:
        test_initialize_affine_weight(tensor_model_parallel_size)
        tensor_model_parallel_size *= 2

    tensor_model_parallel_size = 1
    while tensor_model_parallel_size <= world_size:
        print_separator('test parallel embedding')
        test_parallel_embedding(tensor_model_parallel_size)
        tensor_model_parallel_size *= 2

    print_separator('test column-parallel linear')
    tensor_model_parallel_size = 1
    while tensor_model_parallel_size <= world_size:
        test_column_parallel_linear(tensor_model_parallel_size)
        tensor_model_parallel_size *= 2

    print_separator('test row-parallel linear')
    tensor_model_parallel_size = 1
    while tensor_model_parallel_size <= world_size:
        test_row_parallel_linear(tensor_model_parallel_size)
        tensor_model_parallel_size *= 2

    print_separator('test parallel self-attention')
    tensor_model_parallel_size = 1
    while tensor_model_parallel_size <= world_size:
        test_parallel_self_attention(tensor_model_parallel_size)
        tensor_model_parallel_size *= 2

    print_separator('test parallel transformer')
    tensor_model_parallel_size = 1
    while tensor_model_parallel_size <= world_size:
        test_parallel_transformer_layer(tensor_model_parallel_size)
        tensor_model_parallel_size *= 2
