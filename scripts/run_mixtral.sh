# Copyright (c) 2024 Habana Labs, Ltd. an Intel Company.

#!/bin/bash

# ----------------------------------------------------------------------------
# Mixtral model
# Paper: https://arxiv.org/pdf/2401.04088.pdf
# ----------------------------------------------------------------------------

set -ex

# ----------------------------------------------------------------------------
# User configurable parameters

DATA_DIR=${HL_DATA_DIR_ROOT:-/data/datasets/oscar}
DATA_CACHE_DIR=${HL_DATA_CACHE_DIR:-}
DATA_FILE_PREFIX=${HL_DATA_FILE_PREFIX:-oscar}
TOKENIZER_TYPE=${HL_TOKENIZER_TYPE:-GPTSentencePieceTokenizer}
TOKENIZER_MODEL=${HL_TOKENIZER_MODEL:-}
NUM_NODES=${HL_NUM_NODES:-1}
DP=${HL_DP:-4}
TP=${HL_TP:-2}
PP=${HL_PP:-1}
NO_PIPELINE_PARALLEL=${HL_NO_PIPELINE_PARALLEL:-0}
MICRO_BATCH=${HL_MICRO_BATCH:-1}
EXIT_INTERVAL=${HL_EXIT_INTERVAL:-0}
OUTPUT_DIR=${HL_RESULTS_DIR:-}
OUTPUT_DIR_PREFIX=${HL_RESULTS_DIR_PREFIX:-.}
CHECKPOINT_SAVE=${HL_SAVE:-1}
SAVE_INTERVAL=${HL_SAVE_INTERVAL:-2000}
CHECKPOINTS_DIR=${HL_CHECKPOINTS_DIR:-}
CHECKPOINT_LOAD_TAG=${HL_CHECKPOINT_LOAD_TAG:-}
TENSORBOARD_DIR=${HL_TENSORBOARD_DIR:-}
KILL_SWITCH_FILE=${HL_KILL_SWITCH:-}
HOSTSFILE=${HL_HOSTSFILE:-}
CKP_ACT=${HL_CKP_ACT:-3}
UNIV_CP=${HL_UNIV_CP:-0}
VERIFY_CP=${HL_VERIFY_CP:-0}
QNPU_DIR=${HL_QNPU_DIR:-}
LOG_INTERVAL=${HL_LOG_INTERVAL:-10}
MIXTRAL_MODEL=${HL_MIXTRAL_MODEL:-8x7b}
DEVICES_PER_NODE=${HL_DEVICES_PER_NODE:-8}
ZERO_STAGE=${HL_ZERO_STAGE:-0}
SEQ_PARALLEL=${HL_SEQ_PARALLEL:-0}
OPTIMIZER=${HL_OPTIMIZER:-fusedadamw}
DROPOUT=${HL_DROPOUT:-0.0}
TRAIN_ITERS=${HL_TRAIN_ITERS:-250000}
LR_WARMUP_ITERS=${HL_LR_WARMUP_ITERS:-2000}
EVAL_ITERS=${HL_EVAL_ITERS:-100}
EVAL_INTERVAL=${HL_EVAL_INTERVAL:-1000}
PROFILE=${HL_PROFILE:-} # provide either of pt, pt-full, hltv
PROFILE_STEPS=${HL_PROFILE_STEPS:-"3,4"}
MOE_NUM_CAPACITY_BINS=${HL_MOE_NUM_CAPACITY_BINS:-0}
MOE_CAPACITY_BINS=${HL_MOE_CAPACITY_BINS:-}
MOE_CAPACITY_BINS_EXP_BASE=${HL_CAPACITY_BINS_EXP_BASE:-1.5}
MOE_CAPACITY_BINS_ALIGNMENT=${HL_MOE_CAPACITY_BINS_ALIGNMENT:-64}
MOE_CAPACITY_BINS_OPTIMIZE_INTERVAL=${HL_MOE_CAPACITY_BINS_OPTIMIZE_INTERVAL:-300}
MOE_CAPACITY_BINS_OPTIMIZE_MAX_GROUP=${HL_MOE_CAPACITY_BINS_OPTIMIZE_MAX_GROUP:-4}
MOE_MIN_CAP=${HL_MOE_MIN_CAP:-64}
MOE_ENABLE_EXPERT_TP=${HL_MOE_ENABLE_EXPERT_TP:-0}
MOE_EP=${HL_MOE_EP:-}
MOE_USE_DATA_BEFORE_EXPERT_PARALLEL=${HL_MOE_USE_DATA_BEFORE_EXPERT_PARALLEL:-0}
USE_LAZY_MODE=${HL_USE_LAZY_MODE:-1}
USE_TORCH_COMPILE=${HL_USE_TORCH_COMPILE:-0}
USE_FUSED_SDPA=${HL_USE_FUSED_SDPA:-1}
USE_FUSED_SDPA_WITH_RECOMPUTE=${HL_USE_FUSED_SDPA_WITH_RECOMPUTE:-1}
PARTITIONED_MODE=${HL_PARTITIONED_MODE:-false}
USE_TRANSFORMER_ENGINE=${HL_USE_TRANSFORMER_ENGINE:-0}
USE_CACHE_FP8_WEIGHT=${HL_USE_CACHE_FP8_WEIGHT:-0}
USE_CACHE_FP8_WEIGHT_FWD=${HL_USE_CACHE_FP8_WEIGHT_FWD:-0}
FP8_FORMAT=${HL_FP8_FORMAT:-hybrid} # hybrid or e5m2
GRAD_ACCUM_DTYPE=${HL_GRAD_ACCUM_DTYPE}
FP8_MARGIN=${HL_FP8_MARGIN:-0}
FP8_AMAX_RECOMPUTE_ALGO=${HL_FP8_AMAX_RECOMPUTE_ALGO:-max} # max or most_recent
USE_FUSED_RMSNORM=${HL_USE_FUSED_RMSNORM:-1}
RUN_EVAL_HARNESS=${HL_RUN_EVAL_HARNESS:-0}
EVAL_TASKS=${HL_EVAL_TASKS:-}
TRUST_REMOTE_CODE=${HL_TRUST_REMOTE_CODE:-0}
# Following configuration are dependant on specific model definitions, but can
# be overridden for debug purposes
# - HL_MOE_NUM_EXPERTS
# - HL_NUM_LAYERS
# - HL_SEQ_LEN
# - HL_GBS
# - HL_TRAIN_ITERS

# ----------------------------------------------------------------------------
# Verify supported configuration

if [ $PARTITIONED_MODE != 'false' ]; then
    echo "Currently PipelineEngine does not support partitioning of 2+ outputs from MoE; Configured with HL_PARTITIONED_MODE=${HL_PARTITIONED_MODE}"
    exit 1
fi

if [[ $MOE_ENABLE_EXPERT_TP -eq 0 && $TP -ne 1 ]]; then
    echo "When using TP, MOE must also be configured with TP"
    exit 1
fi

if [ $UNIV_CP -ne 0 ]; then
    echo "No support for loading from universal checkpoint; Configured with HL_UNIV_CP=${HL_UNIV_CP}"
    exit 1
fi

if [ $VERIFY_CP -ne 0 ]; then
    echo "No support for checkpoint verification; Configured with HL_VERIFY_CP=${HL_VERIFY_CP}"
    exit 1
fi

NUM_DEVICES=$(($DP * $TP * $PP))
NUM_DEVICES_2=$(($DEVICES_PER_NODE * $NUM_NODES))
if [ $NUM_DEVICES -ne $NUM_DEVICES_2 ]; then
    echo "Bad devices configuration. DPxTPxPP=${NUM_DEVICES} != N_NODES*N_DEVICES_PER_NODE=${NUM_DEVICES_2}"
    exit 1
fi

# ----------------------------------------------------------------------------
# Mixtral architecture

if [ $MIXTRAL_MODEL == "8x7b" ]; then
    # Mixtral-8x7B model architecture
    MOE_NUM_EXPERTS=${HL_MOE_NUM_EXPERTS:-8}
    N_LAYERS=${HL_NUM_LAYERS:-32}
    SEQ_LEN=${HL_SEQ_LEN:-32768}
    NHIDDEN=4096
    FFN_HIDDEN_SIZE=14336
    NHEADS=32
    NUM_KV_HEADS=8
    LR=3e-4
    MIN_LR=3e-6    # using 0.01 of max-lr (DeepSpeed-MoE https://arxiv.org/pdf/2201.05596.pdf section 3.2)
elif [ $MIXTRAL_MODEL == "small" ]; then
    MOE_NUM_EXPERTS=${HL_MOE_NUM_EXPERTS:-4}
    N_LAYERS=${HL_NUM_LAYERS:-8}
    SEQ_LEN=${HL_SEQ_LEN:-256}
    NHIDDEN=768
    FFN_HIDDEN_SIZE=3072
    NHEADS=16
    NUM_KV_HEADS=8
    LR=3e-4
    MIN_LR=3e-6
else
    echo "Unsupported HL_MIXTRAL_MODEL=$MIXTRAL_MODEL"
    exit 1
fi

if [ -z "${MOE_EP}" ]; then
  if [[ $MOE_NUM_EXPERTS -gt $NUM_DEVICES ]]; then
      MOE_EP=${NUM_DEVICES}
  else
      MOE_EP=${MOE_NUM_EXPERTS}
  fi
fi
echo "Using Num Experts=${MOE_NUM_EXPERTS} with MoE EP=${MOE_EP}"

# ----------------------------------------------------------------------------
# Training configuration: Mixtral paper has no details on training regime.
# Therefore using LLAMA1 regime.
# So, assuming LLAMA1 regime with few exceptions:
# - seq_len = 32768
# - smaller min_lr

TOKENS_IN_BATCH=$((2 ** 22))  # 4M tokens
CALCULATED_GBS=$(($TOKENS_IN_BATCH / $SEQ_LEN))
GLOBAL_BATCH=${HL_GBS:-$CALCULATED_GBS}
TOTAL_TOKENS=$((250000 * $TOKENS_IN_BATCH))  # ~1T tokens

# ----------------------------------------------------------------------------
# PATHs

if [[ -z "$MEGATRON_DEEPSPEED_ROOT" ]]; then
    MEGATRON_DEEPSPEED_ROOT=$(realpath $(dirname $0)/../)
fi

DATA_PATH=${DATA_DIR}/${DATA_FILE_PREFIX}

RUNTIME=`date +"%Y%m%d_%H%M"`
# output paths
if [ -z "$OUTPUT_DIR" ]; then
    # Experiment name
    if [ -z "$EXP_NAME" ]; then
        EXP_NAME="default"
    fi
    OUTPUT_DIR=${OUTPUT_DIR_PREFIX}/out/mixtral_${MIXTRAL_MODEL}/ds_${EXP_NAME}_z${ZERO_STAGE}_nl${N_LAYERS}_hs${NHIDDEN}_ffn${FFN_HIDDEN_SIZE}_moe_exp${MOE_NUM_EXPERTS}_gb${GLOBAL_BATCH}_mb${MICRO_BATCH}_sp${SEQ_PARALLEL}_D${DP}_T${TP}_P${PP}_E${MOE_EP}_moeT${MOE_ENABLE_EXPERT_TP}_devices${NUM_DEVICES}_${RUNTIME}
fi

if [ -z "$CHECKPOINTS_DIR" ]; then
    CHECKPOINTS_DIR=$OUTPUT_DIR/checkpoints
fi

if [ -z "$TENSORBOARD_DIR" ]; then
    TENSORBOARD_DIR=$OUTPUT_DIR/tensorboard
fi

mkdir -p ${OUTPUT_DIR}
mkdir -p ${TENSORBOARD_DIR}

# ----------------------------------------------------------------------------
# Create DS config

if [ $SEQ_PARALLEL -eq 1 ]; then
    PARTITIONED_MODE="false"
fi

# Currently, PipelineEngine does not support partitioning of 2+ outputs that
# require gradients). Therefore, disable partitioned mode if using pipeline
# and MoE experts
if [[ ${MOE_NUM_EXPERTS} -gt 1 ]] && [[ ${PP} -ne 1 ]]; then
    PARTITIONED_MODE="false"
fi

# optional grad_accum_dtype setting
DS_CONFIG_GRAD_ACCUM_DTYPE=""
if [[ -n "$GRAD_ACCUM_DTYPE" ]]; then
    DS_CONFIG_GRAD_ACCUM_DTYPE=",
    \"data_types\": {
    \"grad_accum_dtype\": \"$GRAD_ACCUM_DTYPE\"
  }"
fi

DS_CONFIG=${OUTPUT_DIR}/ds_config.json
cat << EOT > $DS_CONFIG
{
  "train_batch_size" : $GLOBAL_BATCH,
  "train_micro_batch_size_per_gpu": $MICRO_BATCH,
  "steps_per_print": $LOG_INTERVAL,
  "gradient_clipping": 1.0,
  "zero_optimization": {
    "stage": $ZERO_STAGE
  },
  "zero_allow_untested_optimizer": true,
  "bf16": {
    "enabled": true,
    "immediate_grad_update": true
  },
  "fp16": {"enabled": false},
  "wall_clock_breakdown": false,
  "pipeline": {
    "pipe_partitioned": $PARTITIONED_MODE,
    "grad_partitioned": $PARTITIONED_MODE
  },
  "use_data_before_expert_parallelism": $MOE_USE_DATA_BEFORE_EXPERT_PARALLEL
  $DS_CONFIG_GRAD_ACCUM_DTYPE
}
EOT

# ----------------------------------------------------------------------------
# Create command

# configure multi-node
MULTINODE_CMD=""
if [ "$NUM_NODES" -ne "1" -a -f "$HOSTSFILE" ]; then
    MULTINODE_CMD="--hostfile=$HOSTSFILE \
                   --master_addr $(head -n 1 $HOSTSFILE | sed -n s/[[:space:]]slots.*//p) "
fi

# training script command
CMD=""
if [ ! -z "$QNPU_DIR" ]; then
    CMD="source ${QNPU_DIR}/activate ;"
fi

if [ $USE_LAZY_MODE -eq 0 ]; then
    CMD="${CMD} PT_HPU_LAZY_MODE=0"
else
    LOWER_CASE_USE_TORCH_COMPILE=$(echo "$USE_TORCH_COMPILE" | tr '[:upper:]' '[:lower:]')
    if [[ "$LOWER_CASE_USE_TORCH_COMPILE" == "true" || "$LOWER_CASE_USE_TORCH_COMPILE" == "1" ]]; then
        echo "Cannot use lazy(HL_USE_LAZY_MODE) and torch.compile(HL_USE_TORCH_COMPILE) modes together"
        exit 1
    fi
fi

if [ $RUN_EVAL_HARNESS -eq 1 ]; then
    # EVAL_TASKS="wikitext,hellaswag,triviaqa,webqs,winogrande,arc_challenge,arc_easy,openbookqa,race,boolq,cb,copa,rte,wic,wsc,multirc,record,anli_r1,anli_r2,anli_r3"
    # Either "all" or comma separated list of tasks.
    # --adaptive_seq_len is currently not supported due to restraints in dynamic MoE implementation
    MDS_SCRIPT="${MEGATRON_DEEPSPEED_ROOT}/tasks/eval_harness/evaluate.py \
    --task_list ${EVAL_TASKS} \
    --no-load-optim \
    --no-load-rng \
    --inference \
    --eval_fp32 \
    --results_path ${OUTPUT_DIR}/eval_task_result_${RUNTIME}.json"
    GLOBAL_BATCH=$DP # fix GBS to DP to avoid OOM on eval tasks with varying seqlen

else
    MDS_SCRIPT="${MEGATRON_DEEPSPEED_ROOT}/pretrain_gpt.py \
    --deepspeed_config=${DS_CONFIG}"
fi

CMD="${CMD} \
    python3 -u ${MDS_SCRIPT} \
    --bf16 \
    --deepspeed \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --num-layers ${N_LAYERS} \
    --hidden-size ${NHIDDEN} \
    --ffn-hidden-size ${FFN_HIDDEN_SIZE} \
    --num-attention-heads ${NHEADS} \
    --num-key-value-heads ${NUM_KV_HEADS} \
    --seq-length ${SEQ_LEN} \
    --micro-batch-size ${MICRO_BATCH} \
    --global-batch-size ${GLOBAL_BATCH} \
    --train-iters ${TRAIN_ITERS} \
    --log-interval ${LOG_INTERVAL} \
    --eval-iters ${EVAL_ITERS} \
    --eval-interval ${EVAL_INTERVAL} \
    --data-path ${DATA_PATH} \
    --optimizer ${OPTIMIZER} \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --adam-eps 1e-8 \
    --lr ${LR} \
    --min-lr ${MIN_LR} \
    --lr-decay-style cosine \
    --lr-warmup-iters ${LR_WARMUP_ITERS} \
    --clip-grad 1.0 \
    --weight-decay 0.1 \
    --tensorboard-dir ${TENSORBOARD_DIR} \
    --log-validation-ppl-to-tensorboard \
    --log-batch-size-to-tensorboard \
    --log-timers-to-tensorboard \
    --load ${CHECKPOINTS_DIR} \
    --use-torch-compile=${USE_TORCH_COMPILE} \
    --zero-stage=${ZERO_STAGE} \
    --exit-interval ${EXIT_INTERVAL} \
    --no-masked-softmax-fusion \
    --no-bias-gelu-fusion \
    --no-bias-dropout-fusion \
    --no-gradient-accumulation-fusion \
    --max-position-embeddings ${SEQ_LEN} \
    --use-rotary-position-embeddings \
    --rotary-position-embeddings-theta 1000000 \
    --untie-embeddings-and-output-weights \
    --swiglu \
    --normalization rmsnorm \
    --disable-bias-linear \
    --no-query-key-layer-scaling \
    --attention-dropout ${DROPOUT} \
    --hidden-dropout ${DROPOUT} \
    --use-fused-sdpa ${USE_FUSED_SDPA} \
    --use-fused-sdpa-with-recompute ${USE_FUSED_SDPA_WITH_RECOMPUTE} \
    --use-fused-rmsnorm $USE_FUSED_RMSNORM"

# -------------
# MoE arguments
# -------------
MOE_ARGS=" \
    --num-experts ${MOE_NUM_EXPERTS} \
    --moe-expert-parallel-size ${MOE_EP} \
    --topk 2 \
    --disable-moe-token-dropping \
    --moe-loss-coeff 0.02 \
    --expert-interval 1 \
    --moe-train-capacity-factor 1.0 \
    --moe-eval-capacity-factor 1.0 \
    --moe-min-capacity ${MOE_MIN_CAP} \
    "

if [[ $MOE_NUM_EXPERTS -gt 1 ]]; then
    MOE_ARGS="${MOE_ARGS} --create-moe-param-group"
fi

if [[ $MOE_ENABLE_EXPERT_TP -gt 0 ]]; then
    MOE_ARGS="${MOE_ARGS} --enable-expert-tensor-parallelism"
fi

# ---------------------------
# MoE Capacity Bins arguments
# ---------------------------

MOE_CAPACITY_BINS_ARGS=" \
    --moe-num-capacity-bins ${MOE_NUM_CAPACITY_BINS} \
    --moe-capacity-bins-exp-base ${MOE_CAPACITY_BINS_EXP_BASE} \
    --moe-capacity-bins-alignment ${MOE_CAPACITY_BINS_ALIGNMENT} \
    --moe-capacity-bins-optimize-interval ${MOE_CAPACITY_BINS_OPTIMIZE_INTERVAL} \
    --moe-capacity-bins-optimize-max-group ${MOE_CAPACITY_BINS_OPTIMIZE_MAX_GROUP} \
    "

if [ ! -z "$MOE_CAPACITY_BINS" ]; then
    MOE_CAPACITY_BINS_ARGS="${MOE_CAPACITY_BINS_ARGS} --moe-capacity-bins ${MOE_CAPACITY_BINS}"
fi

if [[ $MOE_NUM_CAPACITY_BINS -gt 0 ]]; then
    MOE_ARGS="${MOE_ARGS} ${MOE_CAPACITY_BINS_ARGS}"
fi

CMD="${CMD} ${MOE_ARGS}"

# ---------------------------
# FP8 arguments
# ---------------------------
if [ $USE_TRANSFORMER_ENGINE -eq 1 ]; then
    CMD="${CMD} --transformer-impl transformer_engine"

    if [ $USE_CACHE_FP8_WEIGHT -eq 1 ]; then
        CMD="${CMD} --cache-fp8-weight"
    fi

    FP8_MEASURE_INTERVAL=${HL_FP8_MEASURE_INTERVAL:-$(( GLOBAL_BATCH / MICRO_BATCH / DP ))}
    FP8_AMAX_HISTORY_LEN=${HL_FP8_AMAX_HISTORY_LEN:-$(( GLOBAL_BATCH / MICRO_BATCH / DP ))}
    FP8_AMAX_REDUCE=${HL_FP8_AMAX_REDUCE:-1}

    CMD="${CMD} --cache-fp8-weight-fwd $USE_CACHE_FP8_WEIGHT_FWD"
    CMD="${CMD} --fp8-interval $FP8_MEASURE_INTERVAL"
    CMD="${CMD} --fp8-margin $FP8_MARGIN"
    CMD="${CMD} --fp8-amax-compute-algo $FP8_AMAX_RECOMPUTE_ALGO"
    CMD="${CMD} --fp8-amax-history-len $FP8_AMAX_HISTORY_LEN"

    if [ "$FP8_FORMAT" = "e5m2" ]; then
        CMD="${CMD} --fp8-e5m2"
    else
        CMD="${CMD} --fp8-hybrid"
    fi

    if [ $FP8_AMAX_REDUCE -eq 1 ]; then
        CMD="${CMD} --fp8-amax-reduce"
    fi
fi

# ---------------------------
# Additonal arguments
# ---------------------------

if [ "$TOKENIZER_TYPE" = "GPTSentencePieceTokenizer" ]; then
    CMD="${CMD} --tokenizer-type GPTSentencePieceTokenizer"
    if [[ -z "$TOKENIZER_MODEL" ]]; then
        TOKENIZER_MODEL="${DATA_DIR}/tokenizer.model"
    fi
    CMD="${CMD} --tokenizer-model $TOKENIZER_MODEL"
elif [ "$TOKENIZER_TYPE" = "GPT2BPETokenizer" ]; then
    CMD="${CMD} --tokenizer-type GPT2BPETokenizer"
    CMD="${CMD} --vocab-file $DATA_DIR/gpt2-vocab.json"
    CMD="${CMD} --merge-file $DATA_DIR/gpt2-merges.txt"
else
    echo "incorrect HL_TOKENIZER_TYPE=$TOKENIZER_TYPE is set"
    exit 1
fi

if [ $TRUST_REMOTE_CODE -eq 1 ]; then
    CMD="${CMD} --trust-remote-code"
fi

if [ ! -z "$CHECKPOINT_LOAD_TAG" ]; then
    CMD="${CMD} --load-tag ${CHECKPOINT_LOAD_TAG}"
fi

if [ ! -z "$KILL_SWITCH_FILE" ]; then
    CMD="${CMD} --kill-switch-path $KILL_SWITCH_FILE"
fi

if [ ! -z "$DATA_CACHE_DIR" ]; then
    CMD="${CMD} --data-cache-path ${DATA_CACHE_DIR}"
fi

if [ $SEQ_PARALLEL -eq 1 ]; then
    CMD="${CMD} --sequence-parallel"
fi

if [ $NO_PIPELINE_PARALLEL -eq 1 ]; then
    CMD="${CMD} --no-pipeline-parallel"
fi

if [ $UNIV_CP -eq 1 ]; then
    echo "Loading Universal Checkpoint from ${CHECKPOINTS_DIR}"
    CMD="${CMD} --universal-checkpoint"
fi

if [ $CHECKPOINT_SAVE -eq 1 ]; then
    mkdir -p ${CHECKPOINTS_DIR}
    CMD="${CMD} --save $CHECKPOINTS_DIR --save-interval $SAVE_INTERVAL"

    if [ $VERIFY_CP -eq 1 ]; then
        # TODO: can we use LLaMA model type to verify Mixtral?
        CMD="${CMD} --verify-checkpoint --verify-checkpoint-model-type LLAMA"
    fi
fi

if [ $CKP_ACT -eq 1 ]; then
    CMD="${CMD} --deepspeed-activation-checkpointing --recompute-granularity=full --recompute-method uniform"
elif [ $CKP_ACT -eq 2 ]; then
    CMD="${CMD} --deepspeed-activation-checkpointing --recompute-granularity=selective"
elif [ $CKP_ACT -eq 3 ]; then
    CMD="${CMD} --deepspeed-activation-checkpointing --recompute-granularity=mlp"
elif [ $CKP_ACT -eq 4 ]; then
    CMD="${CMD} --pytorch-activation-checkpointing"
fi

if [ ! -z "$PROFILE" ]; then
    CMD="${CMD} --profile ${PROFILE}"
    CMD="${CMD} --profile-steps ${PROFILE_STEPS}"
fi

if [ ! -z "$QNPU_DIR" ]; then
    rm -rf $HOME/.deepspeed_env
    echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH" >> $HOME/.deepspeed_env
fi

# run!
deepspeed --num_nodes ${NUM_NODES} \
          --num_gpus ${DEVICES_PER_NODE} \
          --no_local_rank \
          --no_python \
          $MULTINODE_CMD \
          /usr/bin/bash -c "$CMD" #2>&1 | tee ${OUTPUT_DIR}/log_${RUNTIME}.txt
