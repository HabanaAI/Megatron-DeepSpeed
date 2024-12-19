# How to run lm-eval on Megatron-DeepSpeed checkpoint
* Follow the instructions on how to setup [here](../../README.md#setup)

# Run MDS Eval Harness

Below example shows running eval harness for LLaMA model.
Need to set `num_gpus, PP, TP, seq_length, tokenizer_model, MBS, GBS, load, load_tag, task` appropriately based on trained Megatron-DeepSpeed checkpoint and it's location. To match lm-eval(HuggingFace checkpoint) way of compute using Megatron-DeepSpeed checkpoint need to use `--attention-softmax-in-fp32`, `--eval-add-bos` and `--eval-hf-rope` command line arguments.

```bash
deepspeed --num_gpus num_gpus $MEGATRON_DEEPSPEED_ROOT/tasks/eval_harness/evaluate.py --pipeline-model-parallel-size PP --tensor-model-parallel-size TP --seq-length seq_length --tokenizer-model /path/to/tokenizer.model --micro-batch-size MBS --global-batch-size GBS --no-load-optim --no-load-rng --no-gradient-accumulation-fusion --bf16 --deepspeed --load /path/to/checkpoint --load-tag /path/to/folder/in/checkpoint/location --inference --eval_fp32 --adaptive_seq_len --use-fused-sdpa 0 --eval-add-bos --task_list task
```

# How to run lm-eval on Hugging Face checkpoint
* Follow the instructions on how to setup Optimum for Intel Gaudi [here](https://github.com/huggingface/optimum-habana/tree/main?tab=readme-ov-file#gaudi-setup)
* Follow the instructions on how to setup lm-eval [here](https://github.com/huggingface/optimum-habana/tree/main/examples/text-generation#lm-eval-requirements)

# Run Eval Harness on Optimum for Intel Gaudi to compare
* Set `num_nodes, num_gpus, BS, task` to match desired running configuration.
* You can choose a number of buckets and values, but the max model sequence length must be accommodated in buckets.

```bash
python -m deepspeed.launcher.runner --num_nodes num_nodes --num_gpus num_gpus --no_local_rank examples/text-generation/run_lm_eval.py --model_name_or_path /path/to/converted/model --batch_size BS --tasks task -o results.txt --warmup 0 --buckets 16 32 64 128 max_model_sequence_length
```
