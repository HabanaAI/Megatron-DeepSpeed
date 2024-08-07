# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company.

#!/bin/bash
DIR=`pwd`
###############################################################################
### Main configs
## GPT-3 models use 2K sequence length/context window
SEQ_LEN=2048

### The "GPT-3 XXX" below are configs from GPT-3 paper
### https://arxiv.org/abs/2005.14165, choose based on
### your desired model size or build your own configs

## GPT-3 Small 125M
MODEL_SIZE=0.125
NUM_LAYERS=12
HIDDEN_SIZE=768
NUM_ATTN_HEADS=12
# GLOBAL_BATCH_SIZE=256
# LR=6.0e-4
MIN_LR=6.0e-5

# Curriculum learning (CL) enables stable large-batch training
GLOBAL_BATCH_SIZE=2048 # 8x
LR=2.4e-3 # 4x

## GPT-3 Medium 350M
# MODEL_SIZE=0.35
# NUM_LAYERS=24
# HIDDEN_SIZE=1024
# NUM_ATTN_HEADS=16
# GLOBAL_BATCH_SIZE=256
# LR=3.0e-4
# MIN_LR=3.0e-5

## GPT-3 Large 760M
# MODEL_SIZE=0.76
# NUM_LAYERS=24
# HIDDEN_SIZE=1536
# NUM_ATTN_HEADS=16
# GLOBAL_BATCH_SIZE=256
# LR=2.5e-4
# MIN_LR=2.5e-5

## GPT-3 XL 1.3B
# MODEL_SIZE=1.3
# NUM_LAYERS=24
# HIDDEN_SIZE=2048
# NUM_ATTN_HEADS=16
# GLOBAL_BATCH_SIZE=512
# LR=2.0e-4
# MIN_LR=2.0e-5

## GPT-3 2.7B
# MODEL_SIZE=2.7
# NUM_LAYERS=32
# HIDDEN_SIZE=2560
# NUM_ATTN_HEADS=32
# GLOBAL_BATCH_SIZE=512
# LR=1.6e-4
# MIN_LR=1.6e-5

## GPT-3 6.7B
# MODEL_SIZE=6.7
# NUM_LAYERS=32
# HIDDEN_SIZE=4096
# NUM_ATTN_HEADS=32
# GLOBAL_BATCH_SIZE=1024
# LR=1.2e-4
# MIN_LR=1.2e-5

## GPT-3 13B
# MODEL_SIZE=13
# NUM_LAYERS=40
# HIDDEN_SIZE=5120
# NUM_ATTN_HEADS=40
# GLOBAL_BATCH_SIZE=1024
# LR=1.0e-4
# MIN_LR=1.0e-5

## GPT-3 175B
# MODEL_SIZE=175
# NUM_LAYERS=96
# HIDDEN_SIZE=12288
# NUM_ATTN_HEADS=96
# GLOBAL_BATCH_SIZE=1536
# LR=0.6e-4
# MIN_LR=0.6e-5
###############################################################################
### Training duration configs
## The main termination condition, original GPT-3 paper trains for 300B tokens
TRAIN_TOKENS=300000000000

## TRAIN_SAMPLES is another termination condition and also affect the number of 
## data samples to be indexed. Since we want to reach the TRAIN_TOKENS
## above, and techniques like curriculum learning has less token in some samples,
## so we just set this config large enough to make sure we have enough
## processed data and don't terminate by TRAIN_SAMPLES.
TRAIN_SAMPLES=$(( ${TRAIN_TOKENS} * 3 / ${SEQ_LEN} ))

## Another termination condition in minutes. Set it large enough to avoid
## undesired early termination.
EXIT_DURATION=30000000
###############################################################################
### LR configs
## LR warmup and decay duration, this token-based config is preferable since
## no need to readjust when the batch size/seqlen is changed.
## Original GPT-3 paper uses 375M warmup tokens and 260B decay tokens.
WARMUP_TOKENS=375000000
LR_DECAY_TOKENS=260000000000
###############################################################################
### Parallelism configs
## Micro batch size per GPU
## Make sure that BATCH_SIZE <= GLOBAL_BATCH_SIZE*PP_SIZE*MP_SIZE/NUM_GPUS
BATCH_SIZE=16

## Model parallelism, 1 is no MP
MP_SIZE=1

## Pipeline parallelism. To disable PP, set PP_SIZE to 1 and NO_PP to true.
PP_SIZE=1
NO_PP="true"

## ZeRO stage
ZERO_STAGE=0

## Total number of GPUs
nvidia-smi || count_GPU=0
if [[ ${count_GPU} == 0 ]];then
    NUM_GPUS=$(lspci | grep -i "Processing accelerators: Habana Labs Ltd." | wc -l)
    NUM_GPUS_PERNODE=${NUM_GPUS}
else
    NUM_GPUS=$(($(ds_ssh nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)-2))
    NUM_GPUS_PERNODE=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
fi
NUM_NODE=$(( ${NUM_GPUS} / ${NUM_GPUS_PERNODE} ))
DP_SIZE=$(( ${NUM_GPUS} / ${PP_SIZE} / ${MP_SIZE} ))
###############################################################################
### Curriculum learning (CL) configs
## Enable/disable CL
CL_ENABLED="true"
## Consult the tutorial https://www.deepspeed.ai/tutorials/curriculum-learning/
## for tuning the following configs
CL_START_SEQLEN=72
CL_AVG_SEQLEN=$(( (${CL_START_SEQLEN} + ${SEQ_LEN}) / 2 ))
CL_TOKENS=60
CL_STEP=$(( ${CL_TOKENS} * 1000000000 / (${GLOBAL_BATCH_SIZE} * ${CL_AVG_SEQLEN}) ))
###############################################################################
### Misc configs
LOG_INTERVAL=10
EVAL_ITERS=10
EVAL_INTERVAL=100
SAVE_INTERVAL=1000
EXIT_INTERVAL=${HL_EXIT_INTERVAL:-0}

## Standard deviation for weight initialization. Usually larger model needs
## lower std. We used a heuristic equation of sqrt(1/3/HIDDEN_SIZE) from the
## MT-NLG 530B work (https://arxiv.org/pdf/2201.11990.pdf)
INIT_STD=0.02

## Activation checkpointing saves GPU memory, but reduces training speed
ACTIVATION_CHECKPOINT="true"
# ACTIVATION_CHECKPOINT="false"

## Whether or not log optimizer states (norms, max abs values) to tensorboard.
## This is not required for training and might save GPU memory when turned off.
LOG_OPTIMIZER_STATE="true"
###############################################################################
### Output and data configs
current_time=$(date "+%Y.%m.%d-%H.%M.%S")
host="${HOSTNAME}"
NAME="gpt3-with-pile-${MODEL_SIZE}B-lr-${LR}-minlr-${MIN_LR}-bs-${GLOBAL_BATCH_SIZE}-gpus-${NUM_GPUS}-zero-${ZERO_STAGE}-mp-${MP_SIZE}-pp-${PP_SIZE}"
if [ "${NO_PP}" = "true" ]; then
    NAME="${NAME}-no_pp"
fi
if [ "${CL_ENABLED}" = "true" ]; then
    NAME="${NAME}-cl-startseqlen-${CL_START_SEQLEN}-step-${CL_STEP}-token-${CL_TOKENS}B"
fi

LOG_PATH="log/"
TENSORBOARD_PATH="tensorboard/${NAME}_${host}_${current_time}"
CHECKPOINT_PATH="/blob/users/conglli/project/gpt3_with_pile/checkpoint/${NAME}"
mkdir -p ${LOG_PATH}
mkdir -p ${TENSORBOARD_PATH}
mkdir -p ${CHECKPOINT_PATH}

#VOCAB_PATH=/data/the_pile_public_merged_nopreprocessing/gpt2-vocab.json
#MERGE_PATH=/data/the_pile_public_merged_nopreprocessing/gpt2-merges.txt
# Public the Pile dataset, can be downloaded at https://mystic.the-eye.eu/public/AI/pile_neox/
# For cluster Azure-EastUS-V100-32GB-4, Lab-RR1-V100
#DATA_PATH=/vc_data_blob/users/conglli/the_pile_public_merged_nopreprocessing/pile_text_document
# For cluster Azure-WestUS3-A100
# DATA_PATH=/blob/data/the_pile_public_merged_nopreprocessing/pile_text_document
BASE_DATA_PATH=${HL_DATA_DIR_ROOT:-/data/bigscience/oscar-en/}
VOCAB_PATH=${BASE_DATA_PATH}/gpt2-vocab.json
MERGE_PATH=${BASE_DATA_PATH}/gpt2-merges.txt
DATA_PATH=${BASE_DATA_PATH}/meg-gpt2_text_document
###############################################################################
data_options=" \
         --vocab-file ${VOCAB_PATH} \
         --merge-file ${MERGE_PATH} \
         --data-path ${DATA_PATH} \
         --data-impl mmap"
        
megatron_options=" \
        --override-opt_param-scheduler \
        --adam-beta1 0.9 \
        --adam-beta2 0.95 \
        --tensor-model-parallel-size ${MP_SIZE} \
        --init-method-std ${INIT_STD} \
        --lr-decay-tokens ${LR_DECAY_TOKENS} \
        --lr-warmup-tokens ${WARMUP_TOKENS} \
        --micro-batch-size ${BATCH_SIZE} \
        --exit-duration-in-mins ${EXIT_DURATION} \
        --global-batch-size ${GLOBAL_BATCH_SIZE} \
        --num-layers ${NUM_LAYERS} \
        --hidden-size ${HIDDEN_SIZE} \
        --num-attention-heads ${NUM_ATTN_HEADS} \
        --seq-length ${SEQ_LEN} \
        --max-position-embeddings ${SEQ_LEN} \
        --train-tokens ${TRAIN_TOKENS} \
        --train-samples ${TRAIN_SAMPLES} \
        --lr ${LR} \
        --min-lr ${MIN_LR} \
        --lr-decay-style cosine \
        --split 98,2,0 \
        --exit-interval ${EXIT_INTERVAL} \
        --log-interval ${LOG_INTERVAL} \
        --eval-interval ${EVAL_INTERVAL} \
        --eval-iters ${EVAL_ITERS} \
        --save-interval ${SAVE_INTERVAL} \
        --weight-decay 0.1 \
        --clip-grad 1.0 \
        --hysteresis 2 \
        --num-workers 0 \
        --fp16 \
        --load ${CHECKPOINT_PATH} \
        --save ${CHECKPOINT_PATH} \
        --tensorboard-queue-size 1 \
        --log-timers-to-tensorboard \
        --log-batch-size-to-tensorboard \
        --log-validation-ppl-to-tensorboard \
        --no-gradient-accumulation-fusion \
        --tensorboard-dir ${TENSORBOARD_PATH}"

if [ "${ACTIVATION_CHECKPOINT}" = "true" ]; then
megatron_options="${megatron_options} \
        --checkpoint-activations --recompute-granularity=full --recompute-method=uniform"
fi

if [ "${LOG_OPTIMIZER_STATE}" = "true" ]; then
megatron_options="${megatron_options} \
        --log-optimizer-states-to-tensorboard"
fi

template_json="ds_config_gpt_TEMPLATE.json"
config_json="ds_config_${NAME}.json"
if [[ $ZERO_STAGE -gt 0 ]]; then
sed "s/CONFIG_BATCH_SIZE/${GLOBAL_BATCH_SIZE}/" ${template_json} \
    | sed "s/CONFIG_MBSIZE/${BATCH_SIZE}/" \
    | sed "s/LOG_INTERVAL/${LOG_INTERVAL}/" \
    | sed "s/ZERO_STAGE/${ZERO_STAGE}/" \
    | sed "s/PRESCALE_GRAD/false/" \
    | sed "s/CONFIG_FP16_ENABLED/true/" \
    | sed "s/CONFIG_BF16_ENABLED/false/" \
    | sed "s/CONFIG_CL_ENABLED/${CL_ENABLED}/" \
    | sed "s/CONFIG_CL_MIN/${CL_START_SEQLEN}/" \
    | sed "s/CONFIG_CL_MAX/${SEQ_LEN}/" \
    | sed "s/CONFIG_CL_DURATION/${CL_STEP}/" \
      > ${config_json}
else
sed "s/CONFIG_BATCH_SIZE/${GLOBAL_BATCH_SIZE}/" ${template_json} \
    | sed "s/CONFIG_MBSIZE/${BATCH_SIZE}/" \
    | sed "s/LOG_INTERVAL/${LOG_INTERVAL}/" \
    | sed "s/ZERO_STAGE/${ZERO_STAGE}/" \
    | sed "s/PRESCALE_GRAD/true/" \
    | sed "s/CONFIG_FP16_ENABLED/true/" \
    | sed "s/CONFIG_BF16_ENABLED/false/" \
    | sed "s/CONFIG_CL_ENABLED/${CL_ENABLED}/" \
    | sed "s/CONFIG_CL_MIN/${CL_START_SEQLEN}/" \
    | sed "s/CONFIG_CL_MAX/${SEQ_LEN}/" \
    | sed "s/CONFIG_CL_DURATION/${CL_STEP}/" \
      > ${config_json}
fi

deepspeed_options=" \
            --deepspeed \
            --deepspeed_config ${config_json} \
            --zero-stage ${ZERO_STAGE} \
            --pipeline-model-parallel-size ${PP_SIZE}"

if [[ "${NO_PP}" = "true" ]]; then
deepspeed_options="${deepspeed_options} \
        --no-pipeline-parallel"
fi

if [ "${ACTIVATION_CHECKPOINT}" = "true" ]; then
deepspeed_options="${deepspeed_options} \
        --deepspeed-activation-checkpointing"
fi

## When saving checkpoint to a storage with cache, their could be consistency
## issue of the pointer to latest checkpoint. Here we find the correct pointer
## and broadcast it to all nodes.
ITERATION_FILE="$CHECKPOINT_PATH/latest_checkpointed_iteration.txt"
ITERATION_FILE_2="$CHECKPOINT_PATH/latest"
ITERATION=0
for (( node = 0; node <= NUM_NODE-1; node++ ))
do
    if $(ssh -q worker-"$node" "test -f \"$ITERATION_FILE\""); then
        LOCAL_ITERATION=$(ssh -q worker-"$node" cat $ITERATION_FILE)
        ITERATION=$(( ${LOCAL_ITERATION} > ${ITERATION} ? ${LOCAL_ITERATION} :  ${ITERATION} ))
    fi
done
if [[ $ITERATION -gt 0 ]]; then
    ITERATION_2="global_step${ITERATION}"
    ds_ssh "echo $ITERATION > $ITERATION_FILE"
    ds_ssh "echo $ITERATION_2 > $ITERATION_FILE_2"
fi

run_cmd="deepspeed ${DIR}/../../pretrain_gpt.py ${megatron_options} ${data_options} ${deepspeed_options} &> ${LOG_PATH}/${NAME}_${host}_${current_time}.log"
echo ${run_cmd}
eval ${run_cmd}
set +x