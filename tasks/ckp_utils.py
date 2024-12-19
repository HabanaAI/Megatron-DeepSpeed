# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company.

import torch
import megatron
import deepspeed
from functools import partial
from megatron.arguments import parse_args
from megatron.global_vars import get_args, set_args
from megatron.initialize import initialize_megatron
from megatron.training import setup_model_and_optimizer
from megatron.core.enums import ModelType
from deepspeed.checkpoint.reshape_utils import get_files, get_files_with_prefix
from deepspeed.checkpoint.deepspeed_checkpoint import ARGS_KEY, MODEL_FILE_PREFIX
from pretrain_gpt import model_provider


def load_args_from_deepspeed_checkpoint(dir: str):
    file_list = get_files(dir)
    mp_rank_files = get_files_with_prefix(file_list, MODEL_FILE_PREFIX)
    sd = torch.load(mp_rank_files[0], map_location=torch.device('cpu'))
    args = sd.get(ARGS_KEY, None)
    return args


def allow_loading_from_hpu_checkpoint():
    """ Allows loading a checkpoint trained on HPU to a system without HPU software stack"""
    def my_rebuild(data, dtype, device, requires_grad):
        device = 'cpu' if 'hpu' in device else device
        tensor = torch.from_numpy(data).to(dtype=dtype, device=device)
        tensor.requires_grad = requires_grad
        return tensor

    torch._utils._rebuild_device_tensor_from_numpy = my_rebuild


def override_args(base_args, override, skip_keys, skip_if_specified_keys):
    for k, v in vars(override).items():
        if k in skip_keys:
            continue
        if k in skip_if_specified_keys and getattr(base_args, k) is not None:
            continue
        setattr(base_args, k, v)


def parse_args_and_setup_megatron(extra_args_provider, pre_init_megatron_fn=None):
    """ Sets up the arguments and initializes the megatron

    Below note was copied from eval_harness/evaluate.py from method load_ds_checkpoint_and_setup_megatron()

    Note(Hesslow):
    The model loading is a bit convoluted.
    We want to parse out the model arguments from the checkpoint and use those to initialize megatron-ds.

    However, megatron-ds expects its arguments on the command line. And at that point we don't know them.

    Instead, we use Jasons way: we load the arguments form the checkpoint and then override _parse_args to
    return whatever args we want.

    If the checkpoint is old, some new arguments may have been introduced and the code will expect these arguments to
    exist. In order to support this we _first_ parse the arguments normally, and then override them with the arguments
    from the checkpoint. Keeping the default-value of newer arguments.
    """

    # avoid printing the arguments, since they will later be overridden.
    _print_args = megatron.arguments._print_args
    megatron.arguments._print_args = lambda *_args, **kwarg: None

    # parse the megatorn args, but wait with initializing megatron as they will be overridden later.
    args = parse_args(extra_args_provider)

    # we set below values as we don't validate args
    args.sequence_parallel = False
    model_parallel_size = args.tensor_model_parallel_size * args.pipeline_model_parallel_size
    args.data_parallel_size = args.world_size // model_parallel_size
    args.eval_micro_batch_size = args.micro_batch_size
    if args.global_batch_size is None:
        args.global_batch_size = args.micro_batch_size * args.data_parallel_size
    if args.weight_decay_incr_style == 'constant':
        assert args.start_weight_decay is None
        assert args.end_weight_decay is None
        args.start_weight_decay = args.weight_decay
        args.end_weight_decay = args.weight_decay
    args.curriculum_learning_legacy = False

    cp_args = load_args_from_deepspeed_checkpoint(args.load)

    # update arguments due to name difference from ckpt
    old_to_new_arg = {"apply_layernorm_weight_plus_one": "apply_layernorm_1p"}
    for key in old_to_new_arg.keys():
        if hasattr(cp_args, key):
            setattr(args, old_to_new_arg[key], getattr(cp_args, key))

    skip_keys = ['world_size', 'rank', 'local_rank', 'device_count', 'micro_batch_size', 'global_batch_size',
                 'batch_size', 'tensorboard_dir', 'deepspeed', 'deepspeed_config', 'deepspeed_configuration',
                 'data_parallel_size', 'pipeline_model_parallel_size', 'tensor_model_parallel_size',
                 'moe_expert_parallel_size', 'moe_token_dropping', 'load', 'rampup_batch_size', 'iteration',
                 'inference', 'bias_dropout_fusion', 'masked_softmax_fusion', 'bias_dropout_fusion',
                 'gradient_accumulation_fusion', 'fp16', 'bf16', 'use_seq_len_plus_one_tokens', 'log_interval',
                 'seq_length', 'max_position_embeddings', 'encoder_seq_length', 'distributed_backend', 'device',
                 'recompute_granularity', 'deepspeed_activation_checkpointing', 'eval_micro_batch_size', 'random_ltd',
                 'use_fused_sdpa', 'use_fused_rmsnorm', 'tokenizer_model', 'attention_dropout', 'hidden_dropout',
                 'attention_softmax_in_fp32', 'eval_hf_rope', 'sequence_parallel', 'eval_add_bos']

    skip_if_specified = ['merge_file', 'vocab_file']

    # allow special handling before arguments override
    if pre_init_megatron_fn is not None:
        pre_init_megatron_fn(args, cp_args, skip_keys, skip_if_specified)

    override_args(args, cp_args, skip_keys, skip_if_specified)

    # stop megatron from reparsing the arguments.
    set_args(args)
    initialize_megatron(allow_parsing=False, allow_validating_args=False)
    torch.distributed.barrier()

    # Initializing megatron will update eg. tokenizer size. Override again.
    override_args(args, cp_args, skip_keys, skip_if_specified)

    # Create minimal deepspeed configuration
    if args.deepspeed_config is None:
        args.deepspeed_config_dict = {
            'train_batch_size': args.global_batch_size,
            'train_micro_batch_size_per_gpu': args.micro_batch_size,
            'bf16': {'enabled': args.bf16},
            'fp16': {'enabled': args.fp16},
            'zero_optimization': {'stage': 0},
        }

    # print final arguments.
    _print_args("arguments", args)

    return args


def load_ds_model(parallel_output=True):
    args = get_args()
    assert args.deepspeed, "load_ds_model() only support DeepSpeed models"

    # Loading pipelined models in deepspeed with different TP than it was originally trained on fails
    # due to a sanity check, that makes sure that all state_dicts that we merge contains attention layers.
    # This, however, is not true for pipelining when we will merge the state_dict for the embeddings
    # which does not contain these attention-specific keys.
    # Deepspeed does however manage to load the model if we just turn off this sanity check.
    deepspeed.runtime.state_dict_factory.MegatronSDLoader.sanity_check = lambda self, ckpt_file_name: None

    if args.deepspeed_config is None:
        args.deepspeed_config_dict = {
            'train_batch_size': args.global_batch_size,
            'train_micro_batch_size_per_gpu': args.micro_batch_size,
            'bf16': {'enabled': args.bf16},
            'fp16': {'enabled': args.fp16},
            'zero_optimization': {'stage': 0},
        }

    cp_path = args.load
    args.load = None
    model_provider_ = partial(model_provider, parallel_output=parallel_output)
    model, _, _ = setup_model_and_optimizer(model_provider_, ModelType.encoder_or_decoder)
    model = model[0]
    zero_enabled = model._config.zero_enabled
    model._config.zero_enabled = False
    _, _ = model.load_checkpoint(cp_path, tag=args.load_tag, load_optimizer_states=False,
                                 load_lr_scheduler_states=False, load_module_only=True)
    model._config.zero_enabled = zero_enabled

    return model
