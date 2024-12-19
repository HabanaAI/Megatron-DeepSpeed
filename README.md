# LLM for PyTorch

This directory provides scripts to train the GPT-based LLaMA and Mixtral models in the Megatron-DeepSpeed repository on Intel® Gaudi® AI accelerators.
Before you get started, make sure to review the [Supported Configurations](#supported-configurations).

## Table of Contents
* [Model Overview](#model-overview)
* [Setup](#setup)
* [Training Script Settings](#training-script-settings)
* [LLaMA Training and Examples](#llama-training-and-examples)
* [Mixtral Training and Examples](#mixtral-training-and-examples)
* [Changelog](#changelog)
* [Known Issues](#known-issues)

## Model Overview
This implementation is based on https://github.com/microsoft/Megatron-DeepSpeed at 7eb36a11b3a9c48ed07b93692ccf22bfb5577f7e.
Megatron ([1](https://arxiv.org/pdf/1909.08053.pdf) and [2](https://arxiv.org/pdf/2104.04473.pdf)) is a large, powerful transformer developed by the Applied Deep Learning Research team at NVIDIA. This repository is for training large transformer language models such as LLaMA at scale. Codebase is capable of efficiently training very large (hundreds of billions of parameters) language models with both model and data parallelism.

### How to use
Users bear sole liability and responsibility to follow and comply with any third party licenses, and Habana Labs disclaims and will bear no liability with respect to users’ use or compliance with third party licenses.
* Third-Party Models
  * In the course of using Megatron-LM, users may choose to download models created and distributed by third parties after reviewing background information about the models and agreeing to the license governing those models.
  * Notice: Intel does not create the content and does not warrant its accuracy or quality. By accessing the third-party content, or using materials trained on or with such content, you are indicating your acceptance of the terms associated with that content and warranting that your use complies with the applicable license.
  * Intel expressly disclaims the accuracy, adequacy, or completeness of any such third-party content, and is not liable for any errors, omissions, or defects in the content, or for any reliance on the content. You agree Intel is not liable for any liability or damages relating to your use of third-party content.
  * Intel’s identification of these resources does not expand or otherwise alter Intel’s applicable published warranties or warranty disclaimers for Intel products or solutions, and you agree that no additional obligations, indemnifications, or liabilities arise from Intel identifying such resources. Intel reserves the right, without notice, to make corrections, enhancements, improvements, and other changes to its materials.
  * The table below contains links to the licenses for certain third-party models and detailed information about the capabilities, limitations, and best practices for those models.

    | Model/Component        | Framework         | Mode                | Detailed Information | License |
    | ---------------------- | ----------------- | ------------------- | -------------------- | ------- |
    | LLaMA 2                | PyTorch           | Pretraining         | [Use Policy](https://github.com/meta-llama/llama-models/blob/main/models/llama2/USE_POLICY.md) | [License](https://github.com/meta-llama/llama-models/blob/main/models/llama2/LICENSE) |
    | LLaMA 3.1              | PyTorch           | Pretraining         | [Use Policy](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/USE_POLICY.md) | [License](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/LICENSE) |
    | Meta LLaMA 3 Tokenizer | PyTorch           | Tokenizer Support   | [Use Policy](https://github.com/meta-llama/llama3/blob/main/USE_POLICY.md) | [License](https://github.com/meta-llama/llama3/blob/main/LICENSE) |

# Setup
Please follow the instructions provided in the [Intel Gaudi Installation Guide](https://docs.habana.ai/en/latest/Installation_Guide/index.html)
to set up the environment including the `$PYTHON` environment variable. To achieve the best performance, please follow the methods outlined in the [Optimizing Training Platform guide](https://docs.habana.ai/en/latest/PyTorch/Model_Optimization_PyTorch/Optimization_in_Training_Platform.html).
The guides will walk you through the process of setting up your system to run the model on Gaudi 2.

## Install Intel Gaudi DeepSpeed
Please follow the instructions provided in the [DeepSpeed Installation Guide](https://docs.habana.ai/en/latest/PyTorch/DeepSpeed/DeepSpeed_User_Guide/DeepSpeed_User_Guide.html#installing-deepspeed-library) to install deepspeed.

## Clone Intel Gaudi Megatron-DeepSpeed
In the docker container, clone this repository and switch to the branch that matches your Intel Gaudi software version.
You can run the [`hl-smi`](https://docs.habana.ai/en/latest/System_Management_Tools_Guide/System_Management_Tools.html#hl-smi-utility-options) utility to determine the Intel Gaudi software version.
```bash
git clone -b [Intel Gaudi software version] https://github.com/HabanaAI/Megatron-DeepSpeed
```

```
export MEGATRON_DEEPSPEED_ROOT=/path/to/Megatron-DeepSpeed
export PYTHONPATH=$MEGATRON_DEEPSPEED_ROOT:$PYTHONPATH
```
## Install Megatron-DeepSpeed Requirements
* In the docker container, go to the Megatron-DeepSpeed directory:
  ```bash
  cd $MEGATRON_DEEPSPEED_ROOT
  ```

* Install the required packages using pip:
Review and accept the [LLaMA 3 tokenizer license conditions](https://github.com/meta-llama/llama3/blob/main/LICENSE) before using it
  ```bash
  pip install -r megatron/core/requirements.txt
  ```

* To run training on more than 128 cards, apply the below configuration changes:
  ```bash
  echo '*    soft nofile  unlimited' >> /etc/security/limits.conf
  echo '*    hard nofile  unlimited' >> /etc/security/limits.conf
  echo 'root soft nofile  unlimited' >> /etc/security/limits.conf
  echo 'root hard nofile  unlimited' >> /etc/security/limits.conf
  ```

## Dataset Preparation
Follow the instructions in https://github.com/bigscience-workshop/bigscience/tree/master/data/oscar to download oscar-en full dataset. Note that the dataset takes around 550G of disk space. This dataset is used for training LLaMA & LLaMA 2.
### Dataset Preparation Example
The below provides the steps required to prepare your dataset. It is based on instructions in https://github.com/bigscience-workshop/bigscience/tree/master/data/oscar.  The dataset in the example is intended to be `zh`
### Step 0 :
```bash
git clone https://github.com/bigscience-workshop/bigscience.git
cd bigscience/data/oscar
# Edit the `oscar-to-jsonl.py` in the list language_subsets and remove the comment on unshuffled_deduplicated_zh and comment out unshuffled_deduplicated_en
vi oscar-to-jsonl.py
```
### Step 1 :
```bash
# -s can be added for subset of data
$PYTHON oscar-to-jsonl.py
```
### Step 2 :
  ```bash
mkdir -p zh
mv oscar*.jsonl zh
cd zh
  ```
### Step 3 :
Use one of the three methods below to tokenize the dataset. You can use any number of workers based on the CPU cores.
*  Tokenize the dataset using GPT2BPETokenizer:
    ```bash
    # download gpt2 vocab and merge files
    wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json
    wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt

    # tokenize individual jsonl files
    # loop count will change based on number of files for a given dataset
    mkdir zh_tokenized
    for i in $(seq 0 4);
    do
      $PYTHON $MEGATRON_DEEPSPEED_ROOT/tools/preprocess_data.py --input oscar-${i}.jsonl --output-prefix zh_tokenized/tokenized${i} --tokenizer-type GPT2BPETokenizer --vocab-file gpt2-vocab.json --merge-file gpt2-merges.txt --append-eod --workers 80
    done
    ```
  * Tokenize the dataset using GPTSentencePieceTokenizer:
    ```bash
    # download tokenizer.model based on model trying to train
    # tokenize individual jsonl files
    # loop count will change based on number of files for a given dataset
    mkdir zh_tokenized
    for i in $(seq 0 4);
    do
      $PYTHON $MEGATRON_DEEPSPEED_ROOT/tools/preprocess_data.py --input oscar-${i}.jsonl --output-prefix zh_tokenized/tokenized${i} --tokenizer-type GPTSentencePieceTokenizer --tokenizer-model /path/to/tokenizer.model --append-eod --workers 80
    done
    ```

  * Tokenize the dataset using HFTokenizer:
    ```bash
    # path to tokenizer can be local directory path and to run custom code from it, trust remote code option(--trust-remote-code) should be passed
    #  or
    # path to tokenizer can be link to huggingface repo model card
    # if huggingface repo model card is a gated repo, Log in using a token from huggingface.co/settings/tokens with below command
    # huggingface-cli login
    # --seq-length value need to be passed explicitly from huggingface repo model card or local directory path which has model_max_length in tokenizer_config.json file

    # tokenize individual jsonl files
    # loop count will change based on number of files for a given dataset
    mkdir zh_tokenized
    for i in $(seq 0 4);
    do
      $PYTHON $MEGATRON_DEEPSPEED_ROOT/tools/preprocess_data.py --input oscar-${i}.jsonl --output-prefix zh_tokenized/tokenized${i} --tokenizer-type HFTokenizer --tokenizer-model /path/to/tokenizer --append-eod --workers 4 --seq-length 1000000000000000019884624838656
    done
    ```
### Step 4 :
 * Multiple tokenized dataset files are merged into a single file using the below method:
    ```bash
    # merge tokenized files
    mkdir zh_tokenized_merged
    $PYTHON $MEGATRON_DEEPSPEED_ROOT/tools/merge_datasets.py --input zh_tokenized --output-prefix zh_tokenized_merged/tokenized_text_document
    # use the tokenized files generated from above command to train
    ```

# Training Script Settings
* Based on the tokenization method, update the tokenizer type:
  ```
  HL_TOKENIZER_TYPE=GPT2BPETokenizer
  ```
* To run custom tokenizer code from local path using HFTokenizer method:
  ```
  HL_TRUST_REMOTE_CODE=1
  ```
* Update data root dir with the path of your choice:
  ```
  HL_DATA_DIR_ROOT=/data/bigscience/oscar-en
  ```
* Update data file prefix(*.bin and *.idx) based on file name in data root dir:
  ```
  HL_DATA_FILE_PREFIX=tokenized_text_document
  ```
* Update tokenizer.model file path if it is not in data root dir, required for any sentence piece based tokenizer:
  ```
  HL_TOKENIZER_MODEL=path/to/tokenizer.model
  ```

Note: For the training commands, make sure to change the IP addresses in hostsfile according to your setup.
`HL_RESULTS_DIR` and `HL_DATA_DIR_ROOT` must be shared writable across all nodes and launchers when running training on more than 8 cards.
The same applies to `HL_CHECKPOINTS_DIR`, `HL_TENSORBOARD_DIR` and `HL_KILL_SWITCH` if specified.
If `HL_DATA_DIR_ROOT` is not writable, then `HL_DATA_CACHE_DIR` must be set to a writable location and
must be shared and accessible across all nodes and launchers when running training on more than 8 cards.


# LLaMA Training and Examples
* Training of LLaMA is based on https://arxiv.org/abs/2302.13971
* Training of LLaMA 2 is based on https://arxiv.org/pdf/2307.09288
* Training of LLaMA 3.1 is based on https://ai.meta.com/research/publications/the-llama-3-herd-of-models/

### Multi-Card Training Examples

* Run LlaMA 3.1 8B on 8 HPUs with BF16 & FP8 precision:

  ```bash
  # Additional flags for FP8, concat with BF16 command
  # Retain default settings for optimal performance.
  HL_USE_CACHE_FP8_WEIGHT_FWD=1 \
  HL_USE_CACHE_FP8_WEIGHT=1 \
  HL_USE_TRANSFORMER_ENGINE=1 \

  # BF16 only
  HL_TOKENIZER_TYPE=Llama3Tokenizer \
  HL_ZERO_STAGE=1 \
  HL_CKP_ACT=2 \
  HL_LLAMA_VER=3.1 \
  HL_DP=2 \
  HL_TP=4 \
  HL_PP=1 \
  HL_LLAMA_MODEL_SIZE=8 \
  $MEGATRON_DEEPSPEED_ROOT/scripts/run_llama.sh
  ```

* Run LlaMA 3.1 70B on 64 HPUs with BF16 & FP8 precision:
  ```bash
  # Additional flags for FP8, concat with BF16 command
  # Retain default settings for optimal performance.
  HL_USE_CACHE_FP8_WEIGHT_FWD=1 \
  HL_USE_CACHE_FP8_WEIGHT=1 \
  HL_USE_TRANSFORMER_ENGINE=1 \

  # BF16 only
  HL_HOSTSFILE=$MEGATRON_DEEPSPEED_ROOT/scripts/hostsfile \
  HL_TOKENIZER_TYPE=Llama3Tokenizer \
  HL_ZERO_STAGE=1 \
  HL_CKP_ACT=2 \
  HL_NUM_NODES=8 \
  HL_LLAMA_VER=3.1 \
  HL_DP=8 \
  HL_TP=8 \
  HL_PP=1 \
  HL_LLAMA_MODEL_SIZE=70 \
  $MEGATRON_DEEPSPEED_ROOT/scripts/run_llama.sh
  ```

* Run LLaMA 2 13B on 8 HPUs with BF16 precision:
  ```bash
  HL_TOKENIZER_TYPE=GPTSentencePieceTokenizer \
  HL_LLAMA_VER=2 \
  HL_LLAMA_MODEL_SIZE=13 \
  HL_NUM_NODES=1 \
  HL_PP=2 \
  HL_TP=2 \
  HL_DP=2 \
  scripts/run_llama.sh
  ```

* Run LLaMA 2 13B on 64 HPUs with BF16 precision:
  ```bash
  HL_TOKENIZER_TYPE=GPTSentencePieceTokenizer \
  HL_HOSTSFILE=scripts/hostsfile \
  HL_NUM_NODES=8 \
  HL_LLAMA_VER=2 \
  HL_LLAMA_MODEL_SIZE=13 \
  HL_PP=2 \
  HL_TP=2 \
  HL_DP=16 \
  scripts/run_llama.sh
  ```

* Run LLaMA 2 70B on 32 HPUs with BF16 precision:
  ```bash
  HL_TOKENIZER_TYPE=GPTSentencePieceTokenizer \
  HL_HOSTSFILE=scripts/hostsfile \
  HL_LLAMA_VER=2 \
  HL_LLAMA_MODEL_SIZE=70 \
  HL_NUM_NODES=4 \
  HL_PP=4 \
  HL_TP=8 \
  HL_DP=1 \
  scripts/run_llama.sh
  ```

LLaMA 2 training supports FP8 precision, which improves model performance. To enable FP8, set `HL_USE_TRANSFORMER_ENGINE=1`. Several FP8 parameters adjust model performance, accuracy, and memory utilization. It is not recommended to change the following default parameters, as they are set optimally:
 - `HL_FP8_FORMAT=hybrid`
 - `HL_FP8_MARGIN=0`
 - `HL_FP8_AMAX_RECOMPUTE_ALGO=max`
 - `HL_FP8_AMAX_REDUCE=1`
 - `HL_FP8_MEASURE_INTERVAL=GBS/micro_batch_size/DP`
 - `HL_FP8_AMAX_HISTORY_LEN=GBS/micro_batch_size/DP`

The below parameter can be added to improve model performance while using FP8. Try adding them if you have enough memory:
 - `HL_USE_CACHE_FP8_WEIGHT_FWD=1`
 - `HL_USE_CACHE_FP8_WEIGHT=1`

* Run LLaMA 2 70B on 32 HPUs with FP8 precision:
  ```bash
  HL_TOKENIZER_TYPE=GPTSentencePieceTokenizer \
  HL_HOSTSFILE=scripts/hostsfile \
  HL_LLAMA_VER=2 \
  HL_LLAMA_MODEL_SIZE=70 \
  HL_NUM_NODES=4 \
  HL_PP=4 \
  HL_TP=8 \
  HL_DP=1 \
  HL_CKP_ACT=0 \
  HL_SEQ_LEN=4096 \
  HL_MICRO_BATCH=1 \
  HL_USE_TRANSFORMER_ENGINE=1 \
  HL_USE_CACHE_FP8_WEIGHT_FWD=1 \
  scripts/run_llama.sh
  ```

* Run LLaMA 2 13B on 16 HPUs with FP8 precision:
  ```bash
  HL_TOKENIZER_TYPE=GPTSentencePieceTokenizer \
  HL_HOSTSFILE=scripts/hostsfile \
  HL_NUM_NODES=2 \
  HL_LLAMA_VER=2 \
  HL_LLAMA_MODEL_SIZE=13 \
  HL_PP=2 \
  HL_TP=2 \
  HL_DP=4 \
  HL_CKP_ACT=2 \
  HL_SEQ_LEN=4096 \
  HL_ZERO_STAGE=1 \
  HL_USE_FAST_SOFTMAX=1 \
  HL_MICRO_BATCH=2 \
  HL_GRAD_ACCUM_DTYPE=bf16 \
  HL_USE_TRANSFORMER_ENGINE=1 \
  HL_USE_CACHE_FP8_WEIGHT_FWD=1 \
  HL_USE_CACHE_FP8_WEIGHT=1 \
  scripts/run_llama.sh
  ```

* Run LLaMA 2 7B on 8 HPUs with FP8 precision:
  ```bash
  HL_TOKENIZER_TYPE=GPTSentencePieceTokenizer \
  HL_LLAMA_MODEL_SIZE=7 \
  HL_NUM_NODES=1 \
  HL_LLAMA_VER=2 \
  HL_LLAMA_MODEL_SIZE=7 \
  HL_PP=1 \
  HL_TP=1 \
  HL_DP=8 \
  HL_CKP_ACT=2 \
  HL_SEQ_LEN=4096 \
  HL_ZERO_STAGE=1 \
  HL_USE_FAST_SOFTMAX=1 \
  HL_MICRO_BATCH=1 \
  HL_GRAD_ACCUM_DTYPE=bf16 \
  HL_USE_TRANSFORMER_ENGINE=1 \
  HL_USE_CACHE_FP8_WEIGHT_FWD=1 \
  HL_USE_CACHE_FP8_WEIGHT=1 \
  scripts/run_llama.sh

# Mixtral Training and Examples
* Training of Mixtral is based on https://arxiv.org/abs/2401.04088

### Multi-Card Training Examples
Configure the following for the Mixtral examples below:
* Set the correct path for `HL_DATA_DIR_ROOT`.
* Set the correct values for `HL_TOKENIZER_TYPE` and `HL_DATA_FILE_PREFIX`.
* Add `HL_DATA_CACHE_DIR` and/or `HL_TOKENIZER_MODEL` if necessary.

Refer to [training script settings](#training-script-settings) for details.

In addition, Capacity Bins functionality was introduced for Mixtral. Capacity bins is
a solution for more performance efficent handling of dynamicity in Mixture of Expert layer.
Expert capacity values are limited to a fixed set of values (defined by bins).
Bins are auto-optimized in given steps intervals, based previous bins usage frequencies.

Capacity bins are configured using following variables:
* `HL_MOE_NUM_CAPACITY_BINS` - Number of bins to be used.
* `HL_CAPACITY_BINS_EXP_BASE` - Exponential base for initialization of capacity bins.
Bins are generated with exponential growing bins width.
Bins that are closer to the start are smaller and thus have less extra non-required capacity.
* `HL_MOE_CAPACITY_BINS_ALIGNMENT` - Every capacity bin value (initialized or optimized)
will be a multiple of this alignment.
* `HL_MOE_CAPACITY_BINS_OPTIMIZE_INTERVAL` - Steps interval for auto-optimization of MoE capacity bins.
* `HL_MOE_CAPACITY_BINS_OPTIMIZE_MAX_GROUP` - Maximum group size of adjacent MoE gates
that their capacity bins are optimized jointly.

Capacity bins functionality is enabled by setting `HL_MOE_NUM_CAPACITY_BINS`.
Recomended configuration is to set `HL_MOE_NUM_CAPACITY_BINS=10`
and leave other parameters as default values.

* Run Mixtral 8x7b on 32 HPUs, Lazy mode, with BF16 precision, sequence length 32k:
  ```
  HL_HOSTSFILE=$MEGATRON_DEEPSPEED_ROOT/scripts/hostsfile \
  HL_MOE_NUM_CAPACITY_BINS=10 \
  HL_NUM_NODES=4 \
  HL_TP=8 \
  HL_MOE_EP=1 \
  HL_SEQ_PARALLEL=1 \
  HL_MOE_ENABLE_EXPERT_TP=1 \
  HL_ZERO_STAGE=1 \
  $MEGATRON_DEEPSPEED_ROOT/scripts/run_mixtral.sh
  ```

# Supported Configurations
| Model          | Mode     | Intel Gaudi software Version | PyTorch Version | Validated on Gaudi 2 | Validated on Gaudi 3 |
| ---------------| -------- | ---------------------------- | --------------- | -------------------- | -------------------- |
| LLaMA 2        | Training | 1.19.0                       | 2.5.1           | :heavy_check_mark:   | :heavy_check_mark:   |
| LLaMA 3.1      | Training | 1.19.0                       | 2.5.1           | :heavy_check_mark:   | :heavy_check_mark:   |
| Mixtral 8x7B   | Training | 1.19.0                       | 2.5.1           | :heavy_check_mark:   |                      |

# Changelog
## 1.18.0
 - Added 'mlp' recompute mode support and enabled it for Mixtral (CKP_ACT=3) with FSDPA recompute.
 - Added support for LM Eval Harness from run_mixtral.sh script [here](examples_deepspeed/MoE/readme_evalharness.md#running-lm-eval-in-mixtral).
 - Added Llama 3.1 support and set as default.
## 1.17.0
 - Added throughput timers configuration to the Deepspeed json config.
 - Rebased Megatron-DeepSpeed repository from [PR#372](https://github.com/microsoft/Megatron-DeepSpeed/pull/372) to [PR#374](https://github.com/microsoft/Megatron-DeepSpeed/pull/374).
 - Added support for Megatron-DeepSpeed Eval Harness tasks. Usage example is available [here](tasks/eval_harness/README.md#run-mds-eval-harness).
 - Added support for full recompute in FP8.
 - Added Lazy mode support for Mixtral.
 - Added Capacity Bins functionality for Mixtral.
 - Added Megatron-DeepSpeed to Hugging Face checkpoint conversion support. Usage example is available [here](./tools/convert_checkpoint/README.md#megatron-deepspeed-to-universal-then-to-hf-transformers).
## 1.16.0
 - Added Mixtral model with Eager and torch.compile modes support. Lazy mode is not supported.
 - Rebased Megatron-DeepSpeed repository from [PR#307](https://github.com/microsoft/Megatron-DeepSpeed/pull/307) to [PR#372](https://github.com/microsoft/Megatron-DeepSpeed/pull/372).
 - Set the LLaMA 2 model as the default.
 - Added support for Zeroshot_gpt tasks using DeepSpeed 3D parallelism.
 - Added support for ALiBi positional embeddings in core attention only.
 - Added support for fast softmax. Currently disabled by default.
 - Added support for accumulation of gradients in BF16. Currently disabled by default.
## 1.15.0
 - Initial release.

### Script Modifications
Major changes done to the original model from [microsoft/Megatron-DeepSpeed](https://github.com/microsoft/Megatron-DeepSpeed/commit/3c5f47563f697702c1e305fa01b7563f54b747fc) repository:
* Changed README file content.
* TFLOPs calculation changed.
* Added HPU FP8 support.
* Flash attention support via FusedSDPA is added for HPU Accelerator.
* Added checkpoint verification.
* Added kill-switch mechanism to gracefully stop training.

# Known Issues
* Only scripts and configurations mentioned in this README are supported and verified.
* There is a known accuracy issue with HL_PP > 1 configurations in LlaMA 3.1 70B training with fp8 precision.
