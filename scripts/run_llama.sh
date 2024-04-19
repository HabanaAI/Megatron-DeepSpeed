# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company.

#!/bin/bash

# -------------------------------------------
# LLaMA 7/13/65 B training script
# Paper: https://arxiv.org/abs/2302.13971
# LLaMA 2 7/13/34/70 B training script
# Paper: https://arxiv.org/pdf/2307.09288.pdf
# -------------------------------------------

# LLaMA   Reference script : ../examples_deepspeed/pretrain_llama_distributed.sh
# LLaMA 2 Reference script : ../examples_deepspeed/pretrain_llama2_distributed.sh

set -ex

# ----------------------
# Configurable parameters

DATASET_NAME=${HL_DATASET_NAME:-redpajama}
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
LLAMA_VER=${HL_LLAMA_VER:-1} # 1 for LLaMA and 2 for LLaMA 2
LLAMA_MODEL_SIZE=${HL_LLAMA_MODEL_SIZE:-13}
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
USE_CACHE_FP8_WEIGHT_FWD=${HL_USE_CACHE_FP8_WEIGHT_FWD:-1}
FP8_MEASURE_INTERVAL=${HL_FP8_MEASURE_INTERVAL:-10}
FP8_FORMAT=${HL_FP8_FORMAT:-e5m2}
# ----------------------

if [ $((NUM_NODES*DEVICES_PER_NODE)) -ne $((DP*TP*PP)) ]; then
    echo "NUM_NODES*DEVICES_PER_NODE != DP*TP*PP"
    exit 1
fi

if [[ -z "$MEGATRON_DEEPSPEED_ROOT" ]]; then
    MEGATRON_DEEPSPEED_ROOT=$(realpath $(dirname $0)/../)
fi

DATA_PATH=${DATA_DIR}/${DATA_FILE_PREFIX}

if [ "$LLAMA_VER" = "1" ]; then
    GLOBAL_BATCH=${HL_GBS:-2048} # microbatches in the pipeline (computed as `GLOBAL_BATCH / (DP * MICRO_BATCH)`) should be divisible by the PP
    SEQ_LEN=${HL_SEQ_LEN:-2048}
    TRAIN_ITERS=${HL_TRAIN_ITERS:-250000}
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
else
    GLOBAL_BATCH=${HL_GBS:-1024} # microbatches in the pipeline (computed as `GLOBAL_BATCH / (DP * MICRO_BATCH)`) should be divisible by the PP
    SEQ_LEN=${HL_SEQ_LEN:-4096}
    TRAIN_ITERS=${HL_TRAIN_ITERS:-500000}
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
  "bf16": {
    "enabled": true,
    "accumulate_grads_via_hooks": true
  },
  "fp16": {"enabled": false},
  "wall_clock_breakdown": false,
  "pipeline": {
    "pipe_partitioned": $PARTITIONED_MODE,
    "grad_partitioned": $PARTITIONED_MODE
  }
}
EOT

# configure multi-node
MULTINODE_CMD=""
if [ "$NUM_NODES" -ne "1" -a -f "$HOSTSFILE" ]; then
    MULTINODE_CMD="--hostfile=$HOSTSFILE \
                   --master_addr $(grep -v '^#' $HOSTSFILE |head -n 1 | sed -n s/[[:space:]]slots.*//p) "
fi

# training script command
CMD=""
if [ ! -z "$QNPU_DIR" ]; then
    CMD="source ${QNPU_DIR}/activate ;"
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
    --adam-eps 1e-8 \
    --lr ${LR} \
    --min-lr ${MIN_LR} \
    --lr-decay-style cosine \
    --lr-warmup-iters 2000 \
    --clip-grad 1.0 \
    --weight-decay 0.1 \
    --tensorboard-dir ${TENSORBOARD_DIR} \
    --log-validation-ppl-to-tensorboard \
    --log-batch-size-to-tensorboard \
    --log-timers-to-tensorboard \
    --load ${CHECKPOINTS_DIR} \
    --deepspeed_config=${DS_CONFIG}  \
    --zero-stage=${ZERO_STAGE} \
    --exit-interval ${EXIT_INTERVAL} \
    --no-masked-softmax-fusion \
    --no-bias-gelu-fusion \
    --no-bias-dropout-fusion \
    --no-gradient-accumulation-fusion \
    --bf16 \
    --max-position-embeddings $SEQ_LEN \
    --use-rotary-position-embeddings \
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

if [ "$DATASET_NAME" = "redpajama" ]; then
    CMD="${CMD} --tokenizer-type GPTSentencePieceTokenizer"
    if [[ -z "$TOKENIZER_MODEL" ]]; then
        TOKENIZER_MODEL="${DATA_DIR}/tokenizer.model"
    fi
    CMD="${CMD} --tokenizer-model $TOKENIZER_MODEL"
elif [ "$DATASET_NAME" = "oscar" ]; then
    CMD="${CMD} --tokenizer-type GPT2BPETokenizer"
    CMD="${CMD} --vocab-file $DATA_DIR/gpt2-vocab.json"
    CMD="${CMD} --merge-file $DATA_DIR/gpt2-merges.txt"
fi

if [ "$LLAMA_VER" = "2" ] && [ $NHEADS -ne $NUM_KV_HEADS ]; then
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

    CMD="${CMD} --cache-fp8-weight-fwd $USE_CACHE_FP8_WEIGHT_FWD"
    CMD="${CMD} --fp8-interval $FP8_MEASURE_INTERVAL"

    if [ "$FP8_FORMAT" = "e5m2" ]; then
        CMD="${CMD} --fp8-e5m2"
    else
        CMD="${CMD} --fp8-hybrid"
    fi
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
