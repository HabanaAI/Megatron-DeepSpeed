# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company.

#!/bin/bash

# -------------------------------------------
# LLaMA 7/13/65 B training script
# Paper: https://arxiv.org/abs/2302.13971
# LLaMA 2 7/13/34/70 B training script
# Paper: https://arxiv.org/pdf/2307.09288.pdf
# LLaMA 3.1 8/70/405 B training script
# Paper: https://ai.meta.com/research/publications/the-llama-3-herd-of-models/
# -------------------------------------------

# LLaMA   Reference script : ../examples_deepspeed/pretrain_llama_distributed.sh
# LLaMA 2 Reference script : ../examples_deepspeed/pretrain_llama2_distributed.sh

set -ex

# ----------------------
# Configurable parameters

TRUST_REMOTE_CODE=${HL_TRUST_REMOTE_CODE:-0}
DATA_DIR=${HL_DATA_DIR_ROOT:-/data/datasets/red_pajama}
DATA_CACHE_DIR=${HL_DATA_CACHE_DIR:-}
DATA_FILE_PREFIX=${HL_DATA_FILE_PREFIX:-redpajama}
TOKENIZER_MODEL=${HL_TOKENIZER_MODEL:-}
NUM_NODES=${HL_NUM_NODES:-1}
DP=${HL_DP:-2}
TP=${HL_TP:-2}
PP=${HL_PP:-2}
MICRO_BATCH=${HL_MICRO_BATCH:-1}
EXIT_INTERVAL=${HL_EXIT_INTERVAL:-0}
OUTPUT_DIR=${HL_RESULTS_DIR:-}
OUTPUT_DIR_PREFIX=${HL_RESULTS_DIR_PREFIX:-.}
CHECKPOINT_SAVE=${HL_SAVE:-1}
SAVE_INTERVAL=${HL_SAVE_INTERVAL:-2000}
CHECKPOINTS_DIR=${HL_CHECKPOINTS_DIR:-}
TENSORBOARD_DIR=${HL_TENSORBOARD_DIR:-}
KILL_SWITCH_FILE=${HL_KILL_SWITCH:-}
HOSTSFILE=${HL_HOSTSFILE:-}
CKP_ACT=${HL_CKP_ACT:-0}
UNIV_CP=${HL_UNIV_CP:-0}
QNPU_DIR=${HL_QNPU_DIR:-}
LOG_INTERVAL=${HL_LOG_INTERVAL:-10}
LLAMA_VER=${HL_LLAMA_VER:-3.1} # 1 for LLaMA, 2 for LLaMA 2 & 3.1 for LLaMA 3.1
LLAMA_MODEL_SIZE=${HL_LLAMA_MODEL_SIZE:-8}
DEVICES_PER_NODE=${HL_DEVICES_PER_NODE:-8}
ZERO_STAGE=${HL_ZERO_STAGE:-0}
SEQ_PARALLEL=${HL_SEQ_PARALLEL:-1}
OPTIMIZER=${HL_OPTIMIZER:-fusedadamw}
DROPOUT=${HL_DROPOUT:-0.0}
EVAL_ITERS=${HL_EVAL_ITERS:-100}
EVAL_INTERVAL=${HL_EVAL_INTERVAL:-1000}
USE_FUSED_SDPA=${HL_USE_FUSED_SDPA:-1}
USE_FUSED_SDPA_WITH_RECOMPUTE=${HL_USE_FUSED_SDPA_WITH_RECOMPUTE:-0}
USE_FUSED_RMSNORM=${HL_USE_FUSED_RMSNORM:-1}
PROFILE=${HL_PROFILE:-} # provide either of pt, pt-full, hltv
PROFILE_STEPS=${HL_PROFILE_STEPS:-"3,4"}
USE_TRANSFORMER_ENGINE=${HL_USE_TRANSFORMER_ENGINE:-0}
USE_CACHE_FP8_WEIGHT=${HL_USE_CACHE_FP8_WEIGHT:-0}
USE_CACHE_FP8_WEIGHT_FWD=${HL_USE_CACHE_FP8_WEIGHT_FWD:-0}
FP8_FORMAT=${HL_FP8_FORMAT:-hybrid} # hybrid or e5m2
GRAD_ACCUM_DTYPE=${HL_GRAD_ACCUM_DTYPE}
FP8_MARGIN=${HL_FP8_MARGIN:-0}
FP8_AMAX_RECOMPUTE_ALGO=${HL_FP8_AMAX_RECOMPUTE_ALGO:-max} # max or most_recent
TENSOR_LOGGER=${HL_TENSOR_LOGGER:-0}
TENSOR_LOGGER_DIR=${HL_TENSOR_LOGGER_DIR:-}
TENSOR_LOGGER_START_ITER=${HL_TENSOR_LOGGER_START_ITER:-0}
TENSOR_LOGGER_END_ITER=${HL_TENSOR_LOGGER_END_ITER:-0}
USE_LAZY_MODE=${HL_USE_LAZY_MODE:-1}
USE_TORCH_COMPILE=${HL_USE_TORCH_COMPILE:-0}
USE_TORCH_COMPILED_AUTOGRAD=${HL_USE_TORCH_COMPILED_AUTOGRAD:-0}
COMPILED_AUTOGRAD_FALLBACK_MODE=${HL_COMPILED_AUTOGRAD_FALLBACK_MODE:-"nothing"}
NO_PIPELINE_PARALLEL=${HL_NO_PIPELINE_PARALLEL:-0}
POSITION_EMBEDDING_TYPE=${HL_POSITION_EMBEDDING_TYPE:-rotary}
USE_FAST_SOFTMAX=${HL_USE_FAST_SOFTMAX:-0}
IMMEDIATE_GRAD_UPDATE=${HL_IMMEDIATE_GRAD_UPDATE:-true}
CHECK_TP_NORM=${HL_CHECK_TP_NORM:-0}
START_CHECK_TP_NORM_ITER=${HL_START_CHECK_TP_NORM_ITER:--1}
END_CHECK_TP_NORM_ITER=${HL_END_CHECK_TP_NORM_ITER:--1}
CHECK_TP_NORM_TYPE=${HL_CHECK_TP_NORM_TYPE:-all} # all, io, wb
# ----------------------

if [ $((NUM_NODES*DEVICES_PER_NODE)) -ne $((DP*TP*PP)) ]; then
    echo "NUM_NODES*DEVICES_PER_NODE != DP*TP*PP"
    exit 1
fi

if [  $(( HL_NUM_LAYERS % PP )) -ne 0  ]; then
    echo 'HL_NUM_LAYERS must be divisible by PP'
    exit 1
fi

if [[ -z "$MEGATRON_DEEPSPEED_ROOT" ]]; then
    MEGATRON_DEEPSPEED_ROOT=$(realpath $(dirname $0)/../)
fi

DATA_PATH=${DATA_DIR}/${DATA_FILE_PREFIX}

if [ "$LLAMA_VER" = "1" ]; then
    TOKENIZER_TYPE=${HL_TOKENIZER_TYPE:-GPTSentencePieceTokenizer}
    GLOBAL_BATCH=${HL_GBS:-2048} # microbatches in the pipeline (computed as `GLOBAL_BATCH / (DP * MICRO_BATCH)`) should be divisible by the PP
    SEQ_LEN=${HL_SEQ_LEN:-2048}
    TRAIN_ITERS=${HL_TRAIN_ITERS:-250000}
    ADAM_EPS=1e-8
    LR_WARMUP_ITERS=2000
    ROTARY_POSITION_EMBEDDINGS_THETA=10000
    if [ $LLAMA_MODEL_SIZE -eq 65 ]; then
        # LLaMA-65B model architecture
        N_LAYERS=${HL_NUM_LAYERS:-80} # must be divisible by PP
        NHIDDEN=8192
        NHEADS=64 # must be divisible by TP
        FFN_HIDDEN_SIZE=22016
        LR=1.5e-4
        MIN_LR=1.5e-5
    elif [ $LLAMA_MODEL_SIZE -eq 13 ]; then
        # LLaMA-13B model architecture
        N_LAYERS=${HL_NUM_LAYERS:-40} # must be divisible by PP
        NHIDDEN=5120
        NHEADS=40 # must be divisible by TP
        FFN_HIDDEN_SIZE=13824
        LR=3e-4
        MIN_LR=3e-5
    elif [ $LLAMA_MODEL_SIZE -eq 7 ]; then
        # LLaMA-7B model architecture
        N_LAYERS=${HL_NUM_LAYERS:-32} # must be divisible by PP
        NHIDDEN=4096
        NHEADS=32 # must be divisible by TP
        FFN_HIDDEN_SIZE=11008
        LR=3e-4
        MIN_LR=3e-5
    else
        echo "incorrect HL_LLAMA_MODEL_SIZE=$LLAMA_MODEL_SIZE is set"
        exit 1
    fi
elif [ "$LLAMA_VER" = "2" ]; then
    TOKENIZER_TYPE=${HL_TOKENIZER_TYPE:-GPTSentencePieceTokenizer}
    GLOBAL_BATCH=${HL_GBS:-1024} # microbatches in the pipeline (computed as `GLOBAL_BATCH / (DP * MICRO_BATCH)`) should be divisible by the PP
    SEQ_LEN=${HL_SEQ_LEN:-4096}
    TRAIN_ITERS=${HL_TRAIN_ITERS:-500000}
    ADAM_EPS=1e-8
    LR_WARMUP_ITERS=2000
    ROTARY_POSITION_EMBEDDINGS_THETA=10000
    if [ $LLAMA_MODEL_SIZE -eq 70 ]; then
        # LLaMA2-70B model architecture
        N_LAYERS=${HL_NUM_LAYERS:-80} # must be divisible by PP
        NHIDDEN=8192
        NHEADS=64 # must be divisible by TP
        NUM_KV_HEADS=$((NHEADS/8)) # must be divisible by TP
        FFN_HIDDEN_SIZE=28672
        LR=1.5e-4
        MIN_LR=1.5e-5
    elif [ $LLAMA_MODEL_SIZE -eq 34 ]; then
        # LLaMA2-34B model architecture
        N_LAYERS=${HL_NUM_LAYERS:-48} # must be divisible by PP
        NHIDDEN=8192
        NHEADS=64 # must be divisible by TP
        NUM_KV_HEADS=$((NHEADS/8)) # must be divisible by TP
        FFN_HIDDEN_SIZE=22016
        LR=1.5e-4
        MIN_LR=1.5e-5
    elif [ $LLAMA_MODEL_SIZE -eq 13 ]; then
        # LLaMA2-13B model architecture
        N_LAYERS=${HL_NUM_LAYERS:-40} # must be divisible by PP
        NHIDDEN=5120
        NHEADS=40 # must be divisible by TP
        NUM_KV_HEADS=${NHEADS} # must be divisible by TP
        FFN_HIDDEN_SIZE=13824
        LR=3e-4
        MIN_LR=3e-5
    elif [ $LLAMA_MODEL_SIZE -eq 7 ]; then
        # LLaMA2-7B model architecture
        N_LAYERS=${HL_NUM_LAYERS:-32} # must be divisible by PP
        NHIDDEN=4096
        NHEADS=32 # must be divisible by TP
        NUM_KV_HEADS=${NHEADS} # must be divisible by TP
        FFN_HIDDEN_SIZE=11008
        LR=3e-4
        MIN_LR=3e-5
    else
        echo "incorrect HL_LLAMA_MODEL_SIZE=$LLAMA_MODEL_SIZE is set"
        exit 1
    fi
elif [ "$LLAMA_VER" = "3.1" ]; then
    TOKENIZER_TYPE=${HL_TOKENIZER_TYPE:-Llama3Tokenizer}
    GLOBAL_BATCH=${HL_GBS:-2048} # microbatches in the pipeline (computed as `GLOBAL_BATCH / (DP * MICRO_BATCH)`) should be divisible by the PP
    SEQ_LEN=${HL_SEQ_LEN:-8192}
    TRAIN_ITERS=${HL_TRAIN_ITERS:-937500}
    ADAM_EPS=1e-5
    LR_WARMUP_ITERS=8000
    ROTARY_POSITION_EMBEDDINGS_THETA=500000
    if [ $LLAMA_MODEL_SIZE -eq 8 ]; then
        # LLaMA3.1-8B model architecture
        N_LAYERS=${HL_NUM_LAYERS:-32} # must be divisible by PP
        NHIDDEN=4096
        NHEADS=32 # must be divisible by TP
        NUM_KV_HEADS=8 # must be divisible by TP
        FFN_HIDDEN_SIZE=14336
        LR=3e-4
        MIN_LR=3e-6
    elif [ $LLAMA_MODEL_SIZE -eq 70 ]; then
        # LLaMA3.1-70B model architecture
        N_LAYERS=${HL_NUM_LAYERS:-80} # must be divisible by PP
        NHIDDEN=8192
        NHEADS=64 # must be divisible by TP
        NUM_KV_HEADS=8 # must be divisible by TP
        FFN_HIDDEN_SIZE=28672
        LR=1.5e-4
        MIN_LR=1.5e-6
    elif [ $LLAMA_MODEL_SIZE -eq 405 ]; then
        # LLaMA3.1-405B model architecture
        N_LAYERS=${HL_NUM_LAYERS:-126} # must be divisible by PP
        NHIDDEN=16384
        NHEADS=128 # must be divisible by TP
        NUM_KV_HEADS=8 # must be divisible by TP
        FFN_HIDDEN_SIZE=53248
        LR=8e-5
        MIN_LR=8e-7
    else
        echo "incorrect HL_LLAMA_MODEL_SIZE=$LLAMA_MODEL_SIZE is set"
        exit 1
    fi
else
    echo "incorrect LLAMA_VER=$LLAMA_VER"
    exit 1
fi

RUNTIME=`date +"%Y%m%d_%H%M"`
# output paths
if [ -z "$OUTPUT_DIR" ]; then
    NUM_DEVICES=$(($DP * $TP * $PP))
    # Experiment name
    if [ -z "$EXP_NAME" ]; then
        EXP_NAME="default"
    fi
    OUTPUT_DIR=${OUTPUT_DIR_PREFIX}/out/llama${LLAMA_VER}_${LLAMA_MODEL_SIZE}b/ds_${EXP_NAME}_z${ZERO_STAGE}_nl${N_LAYERS}_hs${NHIDDEN}_ffn${FFN_HIDDEN_SIZE}_gb${GLOBAL_BATCH}_mb${MICRO_BATCH}_sp${SEQ_PARALLEL}_D${DP}_T${TP}_P${PP}_devices${NUM_DEVICES}_${RUNTIME}
fi

if [ -z "$CHECKPOINTS_DIR" ]; then
    CHECKPOINTS_DIR=$OUTPUT_DIR/checkpoints
fi

# if [ $UNIV_CP -eq 1 ]
# then
#     ckpt_name=$(cat $CHECKPOINTS_DIR/latest)
#     res=$(HL_NUM_NODES=$NUM_NODES HL_DEVICES_PER_NODE=$DEVICES_PER_NODE HL_LATEST_CHECKPOINT=$CHECKPOINTS_DIR/$ckpt_name $MEGATRON_DEEPSPEED_ROOT/scripts/convert_ds_to_universal.sh)
# fi

if [ -z "$TENSORBOARD_DIR" ]; then
    TENSORBOARD_DIR=$OUTPUT_DIR/tensorboard
fi

mkdir -p ${OUTPUT_DIR}
mkdir -p ${TENSORBOARD_DIR}

PARTITIONED_MODE="true"
if [ $SEQ_PARALLEL -eq 1 ]; then
    PARTITIONED_MODE="false"
fi

# create DS config

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
    "immediate_grad_update": $IMMEDIATE_GRAD_UPDATE
  },
  "fp16": {"enabled": false},
  "wall_clock_breakdown": false,
  "pipeline": {
    "pipe_partitioned": $PARTITIONED_MODE,
    "grad_partitioned": $PARTITIONED_MODE
  },
  "timers": {
    "throughput": {
      "enabled": true,
      "synchronized": false
    }
  }$DS_CONFIG_GRAD_ACCUM_DTYPE
}
EOT

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
    LOWER_CASE_USE_TORCH_COMPILED_AUTOGRAD=$(echo "$USE_TORCH_COMPILED_AUTOGRAD" | tr '[:upper:]' '[:lower:]')
    if [[ "$LOWER_CASE_USE_TORCH_COMPILED_AUTOGRAD" == "true" || "$LOWER_CASE_USE_TORCH_COMPILED_AUTOGRAD" == "1" ]]; then
        echo "Cannot use lazy(HL_USE_LAZY_MODE) and torch.compiled_autograd(HL_USE_TORCH_COMPILED_AUTOGRAD) modes together"
        exit 1
    fi
fi

CMD="${CMD} \
    python3 -u ${MEGATRON_DEEPSPEED_ROOT}/pretrain_gpt.py \
    --deepspeed \
    --tensor-model-parallel-size $TP \
    --pipeline-model-parallel-size $PP \
    --num-layers ${N_LAYERS} \
    --hidden-size ${NHIDDEN} \
    --ffn-hidden-size ${FFN_HIDDEN_SIZE} \
    --num-attention-heads ${NHEADS} \
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
    --adam-eps ${ADAM_EPS} \
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
    --deepspeed_config=${DS_CONFIG} \
    --use-torch-compile=${USE_TORCH_COMPILE} \
    --use-torch-compiled-autograd=${USE_TORCH_COMPILED_AUTOGRAD} \
    --compiled-autograd-fallback-mode=${COMPILED_AUTOGRAD_FALLBACK_MODE} \
    --zero-stage=${ZERO_STAGE} \
    --exit-interval ${EXIT_INTERVAL} \
    --no-masked-softmax-fusion \
    --no-bias-gelu-fusion \
    --no-bias-dropout-fusion \
    --no-gradient-accumulation-fusion \
    --bf16 \
    --max-position-embeddings $SEQ_LEN \
    --untie-embeddings-and-output-weights \
    --swiglu \
    --normalization rmsnorm \
    --disable-bias-linear \
    --no-query-key-layer-scaling \
    --attention-dropout ${DROPOUT} \
    --hidden-dropout ${DROPOUT} \
    --use-fused-sdpa $USE_FUSED_SDPA \
    --use-fused-sdpa-with-recompute $USE_FUSED_SDPA_WITH_RECOMPUTE \
    --use-fused-rmsnorm $USE_FUSED_RMSNORM"

if [ "$POSITION_EMBEDDING_TYPE" = "rotary" ]; then
    CMD="${CMD} --use-rotary-position-embeddings"
    CMD="${CMD} --rotary-position-embeddings-theta ${ROTARY_POSITION_EMBEDDINGS_THETA}"
else
    CMD="${CMD} --use-alibi-position-embeddings"
fi

if [[ "$TOKENIZER_TYPE" = "GPTSentencePieceTokenizer" || "$TOKENIZER_TYPE" = "Llama3Tokenizer" ]]; then
    CMD="${CMD} --tokenizer-type $TOKENIZER_TYPE"
    if [[ -z "$TOKENIZER_MODEL" ]]; then
        TOKENIZER_MODEL="${DATA_DIR}/tokenizer.model"
    fi
    CMD="${CMD} --tokenizer-model $TOKENIZER_MODEL"
elif [ "$TOKENIZER_TYPE" = "GPT2BPETokenizer" ]; then
    CMD="${CMD} --tokenizer-type GPT2BPETokenizer"
    CMD="${CMD} --vocab-file $DATA_DIR/gpt2-vocab.json"
    CMD="${CMD} --merge-file $DATA_DIR/gpt2-merges.txt"
elif [ "$TOKENIZER_TYPE" = "HFTokenizer" ]; then
    CMD="${CMD} --tokenizer-type HFTokenizer"
    if [[ -z "$TOKENIZER_MODEL" ]]; then
        echo "HL_TOKENIZER_MODEL path is not set"
        exit 1
    fi
    CMD="${CMD} --tokenizer-model $TOKENIZER_MODEL"
    if [ $TRUST_REMOTE_CODE -eq 1 ]; then
        CMD="${CMD} --trust-remote-code"
    fi
else
    echo "incorrect HL_TOKENIZER_TYPE=$TOKENIZER_TYPE is set"
    exit 1
fi

# Enable GQA for LLaMA 2 & 3.1
if [ "$LLAMA_VER" != "1" ] && [ $NHEADS -ne $NUM_KV_HEADS ]; then
    CMD="${CMD} --num-key-value-heads $NUM_KV_HEADS"
fi

if [ ! -z "$DATA_CACHE_DIR" ]; then
    CMD="${CMD} --data-cache-path ${DATA_CACHE_DIR}"
fi

# handle kill switch argument
if [ ! -z "$KILL_SWITCH_FILE" ]; then
    CMD="${CMD} --kill-switch-path $KILL_SWITCH_FILE"
fi

if [ $SEQ_PARALLEL -eq 1 ]
then
    CMD="${CMD} --sequence-parallel"
fi

if [ $USE_FAST_SOFTMAX -eq 1 ]
then
    CMD="${CMD} --use-fast-softmax"
fi

if [ $UNIV_CP -eq 1 ]
then
    echo "Loading Universal Checkpoint from ${CHECKPOINTS_DIR}"
    CMD="${CMD} --universal-checkpoint"
fi

# fp8 args
if [ $USE_TRANSFORMER_ENGINE -eq 1 ]; then
    CMD="${CMD} --transformer-impl transformer_engine"

    if [ $USE_CACHE_FP8_WEIGHT -eq 1 ]; then
        CMD="${CMD} --cache-fp8-weight"
    fi

    FP8_MEASURE_INTERVAL=${HL_FP8_MEASURE_INTERVAL:-$(( GLOBAL_BATCH / MICRO_BATCH / DP ))}
    FP8_AMAX_HISTORY_LEN=${HL_FP8_AMAX_HISTORY_LEN:-$(( GLOBAL_BATCH / MICRO_BATCH / DP ))}
    FP8_AMAX_REDUCE=${HL_FP8_AMAX_REDUCE:-1}
    FP8_MANUAL_MODE=${HL_FP8_MANUAL_MODE:-1}

    CMD="${CMD} --fp8-manual-mode $FP8_MANUAL_MODE"
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

if [[ "$NO_PIPELINE_PARALLEL" == "1" ]]; then
    CMD="${CMD} --no-pipeline-parallel"
fi

if [ $CHECKPOINT_SAVE -eq 1 ]
then
    mkdir -p ${CHECKPOINTS_DIR}
    CMD="${CMD} --save $CHECKPOINTS_DIR --save-interval $SAVE_INTERVAL --verify-checkpoint --verify-checkpoint-model-type LLAMA"
fi

if [ $CKP_ACT -eq 1 ]
then
    CMD="${CMD} --deepspeed-activation-checkpointing --recompute-granularity=full --recompute-method uniform"
elif [ $CKP_ACT -eq 2 ]
then
    CMD="${CMD} --deepspeed-activation-checkpointing --recompute-granularity=selective"
elif [ $CKP_ACT -eq 4 ]
then
    CMD="${CMD} --pytorch-activation-checkpointing"
fi

if [ $TENSOR_LOGGER -eq 1 ]; then
    if [ -z "$TENSOR_LOGGER_DIR" ]; then
        TENSOR_LOGGER_DIR=$OUTPUT_DIR/tensordumps
    fi
    mkdir -p $TENSOR_LOGGER_DIR
    CMD="${CMD} --log-model-inputs"
    CMD="${CMD} --log-fwd-activations"
    CMD="${CMD} --log-bwd-grads"
    CMD="${CMD} --tensor-logger-start-iter $TENSOR_LOGGER_START_ITER"
    CMD="${CMD} --tensor-logger-end-iter $TENSOR_LOGGER_END_ITER"
    CMD="${CMD} --tensor-logger-path $TENSOR_LOGGER_DIR"
fi

if [ $CHECK_TP_NORM -eq 1 ]; then
    CMD="${CMD} --check-tp-norm"
    CMD="${CMD} --start-check-tp-norm-iter $START_CHECK_TP_NORM_ITER"
    CMD="${CMD} --end-check-tp-norm-iter $END_CHECK_TP_NORM_ITER"
    CMD="${CMD} --check-tp-norm-type $CHECK_TP_NORM_TYPE"
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
