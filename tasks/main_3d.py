# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company.
# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Main 3D parallel tasks functionality."""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))
from main import get_tasks_args
from tasks.ckp_utils import allow_loading_from_hpu_checkpoint, parse_args_and_setup_megatron


if __name__ == '__main__':
    allow_loading_from_hpu_checkpoint()

    def pre_init_megatron_fn(args_, _cp_args, skip_keys, _skip_if_specified):
        if args_.checkpoint_override_tokenizer:
            skip_keys += ['merge_file', 'tokenizer_model', 'tokenizer_type',
                          'vocab_extra_ids', 'vocab_file']

    args = parse_args_and_setup_megatron(get_tasks_args, pre_init_megatron_fn)

    if args.task in ['LAMBADA', 'WIKITEXT103']:
        from zeroshot_gpt.evaluate import main
    else:
        raise NotImplementedError(f'Task {args.task} is not implemented.')

    main()
