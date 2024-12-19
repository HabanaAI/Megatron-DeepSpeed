# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company.

#!/bin/bash

# Usage:
# set HL_TRAIN_ITERS for the total iteration to run in training.
# set LLAMA_VER_DIR according to HL_LLAMA_VER and HL_LLAMA_MODEL_SIZE, e.g LLAMA_VER_DIR='llama3_8b'
# make sure to include sheduler-batch-size[0] at the start of the run in batch-sched-samples[0]=consumed samples so far (0 for first run)
# run_llama_seq_sched.sh sheduler-batch-size batch-sched-samples stop-after-num-samples seq-length

# defaults for Llama3.1
# chagne LLAMA_VER_DIR based on your model size
LLAMA_VER_DIR='llama3.1_8b'
# Start with GLOBAL_BATCH_SIZE 1024 and SEQ_LEN 4096 and train for 252M tokens which are 61524 samples (60 iter)
# Tokens to samples conversion:  252M/4096=61524
HL_SCHEDULER_BATCH_SIZE=1024
HL_BATCH_SCHED_SAMPLES=0
HL_STOP_AFTER_NUM_SAMPLES=61524
HL_SEQ_LEN=4096
source scripts/run_llama.sh
# the name of the directory will be determined at runtime based on run params and running time
SEQ_SCHED_CHECKPOINT_DIR=$(ls -dt ./out/$LLAMA_VER_DIR/* | head -n 1)/checkpoints
# After 252M tokens change SEQ_LEN to 8192 and train until 2.87T tokens which are 350311035 samples (342100 iter)
# Tokens to samples conversion:  (2.87T-252M)/8192=350311035
HL_SCHEDULER_BATCH_SIZE=1024
HL_BATCH_SCHED_SAMPLES=61524
HL_STOP_AFTER_NUM_SAMPLES=350311035
HL_SEQ_LEN=8192
HL_CHECKPOINTS_DIR=$SEQ_SCHED_CHECKPOINT_DIR
source  scripts/run_llama.sh
# After 2.87T tokens change GLOBAL_BATCH_SIZE to 2048, and run until training is completed (15000000000000 will not be reached).
HL_SCHEDULER_BATCH_SIZE=2048
HL_BATCH_SCHED_SAMPLES=350311035
HL_STOP_AFTER_NUM_SAMPLES=15000000000000
HL_SEQ_LEN=8192
HL_CHECKPOINTS_DIR=$SEQ_SCHED_CHECKPOINT_DIR
source scripts/run_llama.sh
