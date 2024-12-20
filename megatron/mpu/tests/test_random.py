# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company.
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

from .commons import print_separator
from .commons import initialize_distributed
from megatron.core.tensor_parallel import random
from megatron.core import mpu
import os
import pytest
import torch
from deepspeed.accelerator import get_accelerator
import sys
sys.path.append("../..")


@pytest.fixture
def tensor_model_parallel_size():
    return int(os.getenv("WORLD_SIZE", '1'))


def test_set_cuda_rng_state(tensor_model_parallel_size):

    initialize_distributed()
    if torch.distributed.get_rank() == 0:
        print('> testing set_rng_state with size {} ...'.
              format(tensor_model_parallel_size))

    mpu.initialize_model_parallel(tensor_model_parallel_size)
    tensor_model_parallel_size = mpu.get_tensor_model_parallel_world_size()

    size = 123
    seed = 1234
    get_accelerator().manual_seed(1234)
    tensor = get_accelerator().FloatTensor(size)

    # Get the state
    rng_state = get_accelerator().get_rng_state()
    rng_state_copy = rng_state.clone()

    # Do some stuff.
    for _ in range(5):
        torch.randn(size, out=tensor)
    result_1 = tensor.clone()

    assert rng_state.sub(rng_state_copy).max() == 0
    assert get_accelerator().get_rng_state().sub(rng_state_copy).max() > 0

    # State should be different.
    new_rng_state = get_accelerator().get_rng_state()
    max_diff = new_rng_state.sub(rng_state).max()
    print('   max diff in rng state (should be non-zero) on global rank {}: {}'.
          format(torch.distributed.get_rank(), max_diff))
    assert max_diff > 0

    # Reset the rng state and do the same stuff.
    random._set_cuda_rng_state(rng_state)
    for _ in range(5):
        torch.randn(size, out=tensor)
    random._set_cuda_rng_state(rng_state)
    for _ in range(5):
        torch.randn(size, out=tensor)
    result_2 = tensor.clone()

    # Results should be the same
    error = result_2.sub(result_1).abs().max()
    print('   max error in generated tensors (should be zero) on '
          'global rank {}: {}'.format(torch.distributed.get_rank(), error))
    assert error < 1.0e-6

    # Input state should have remained intact.
    error = rng_state.sub(rng_state_copy).max()
    print('   max error in rng state (should be zero) on global rank {}: {}'.
          format(torch.distributed.get_rank(), error))
    assert error == 0

    # Reset groups
    mpu.destroy_model_parallel()

    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        print('>> passed the test :-)')


def test_cuda_rng_tracker(tensor_model_parallel_size):

    initialize_distributed()
    if torch.distributed.get_rank() == 0:
        print('> testing cuda rng tracker with size {} ...'.
              format(tensor_model_parallel_size))

    mpu.initialize_model_parallel(tensor_model_parallel_size)
    tensor_model_parallel_size = mpu.get_tensor_model_parallel_world_size()

    seed_1 = 1234
    seed_2 = 4321
    size = [12, 21]
    tensor = get_accelerator().FloatTensor(size)

    # Set to seed_1 and generate two tensors.
    get_accelerator().manual_seed(seed_1)
    torch.randn(size, out=tensor)
    target_11 = tensor.clone()
    torch.randn(size, out=tensor)
    target_12 = tensor.clone()

    # Set to seed_2 and generate two tensors.
    get_accelerator().manual_seed(seed_2)
    torch.randn(size, out=tensor)
    target_21 = tensor.clone()
    torch.randn(size, out=tensor)
    target_22 = tensor.clone()

    # Now if we interleave seed_1 and seed_2,
    # we should still get the same tensors
    get_accelerator().manual_seed(seed_1)
    random.get_cuda_rng_tracker().add('test', seed_2)

    torch.randn(size, out=tensor)
    result_11 = tensor.clone()

    with random.get_cuda_rng_tracker().fork('test'):
        torch.randn(size, out=tensor)
        result_21 = tensor.clone()

    torch.randn(size, out=tensor)
    result_12 = tensor.clone()

    with random.get_cuda_rng_tracker().fork('test'):
        torch.randn(size, out=tensor)
        result_22 = tensor.clone()

    diff = result_11.sub(result_21).abs().max()
    diff = min(diff, result_12.sub(result_22).abs().max())
    print('   max diff in generated tensors (should be non-zero) on '
          'global rank {}: {}'.format(torch.distributed.get_rank(), diff))
    assert diff > 1.0e-6
    error = max(result_11.sub(target_11).abs().max(),
                result_12.sub(target_12).abs().max())
    error = max(error, result_21.sub(target_21).abs().max())
    error = max(error, result_22.sub(target_22).abs().max())
    print('   max error in generated tensors (should be zero) on '
          'global rank {}: {}'.format(torch.distributed.get_rank(), error))
    assert error < 1.0e-6

    # Reset the tracker
    random.get_cuda_rng_tracker().reset()

    # Reset groups
    mpu.destroy_model_parallel()

    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        print('>> passed the test :-)')


def test_model_parallel_cuda_manual_seed(tensor_model_parallel_size):

    initialize_distributed()
    if torch.distributed.get_rank() == 0:
        print('> testing model parallel cuda manual seed with size {} ...'.
              format(tensor_model_parallel_size))

    mpu.initialize_model_parallel(tensor_model_parallel_size)
    tensor_model_parallel_size = mpu.get_tensor_model_parallel_world_size()

    random.model_parallel_cuda_manual_seed(12345)
    assert get_accelerator().initial_seed() == 12345
    with random.get_cuda_rng_tracker().fork():
        assert get_accelerator().initial_seed() == (12345 + 2718 +
                                             mpu.get_tensor_model_parallel_rank())

    # Reset the tracker
    random.get_cuda_rng_tracker().reset()

    # Reset groups
    mpu.destroy_model_parallel()

    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        print('>> passed the test :-)')


if __name__ == '__main__':

    initialize_distributed()
    world_size = torch.distributed.get_world_size()

    tensor_model_parallel_size = 1
    while tensor_model_parallel_size <= world_size:
        print_separator('test set rng state')
        test_set_cuda_rng_state(tensor_model_parallel_size)
        tensor_model_parallel_size *= 2

    tensor_model_parallel_size = 1
    while tensor_model_parallel_size <= world_size:
        print_separator('test cuda rng tracker')
        test_cuda_rng_tracker(tensor_model_parallel_size)
        tensor_model_parallel_size *= 2

    tensor_model_parallel_size = 1
    while tensor_model_parallel_size <= world_size:
        print_separator('test model parallel cuda manual seed')
        test_model_parallel_cuda_manual_seed(tensor_model_parallel_size)
        tensor_model_parallel_size *= 2
