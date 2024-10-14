# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/bin/bash

echo "**************************************************************************"
echo "Script to convert Megatron-DeepSpeed Checkpoint into a Universal Checkpoint"
echo "**************************************************************************"

###### INPUTS : START #####
# here for testing only
# MEGATRON_DEEPSPEED_ROOT
# DEEPSPEED_ROOT
# HL_LATEST_CHECKPOINT=<>/checkpoints/global_step48600
# HL_UNIV_CP_EXTRACT_WORKERS
# HL_UNIV_CP_MERGE_WORKERS
###### INPUTS : END  ######

LATEST_CHECKPOINT=${HL_LATEST_CHECKPOINT:-}
EXTRACT_WORKERS=${HL_UNIV_CP_EXTRACT_WORKERS:-}
MERGE_WORKERS=${HL_UNIV_CP_MERGE_WORKERS:-}

if [[ -z "$MEGATRON_DEEPSPEED_ROOT" ]]; then
    MEGATRON_DEEPSPEED_ROOT=$(realpath $(dirname $0)/../)
fi

if [[ -z "$DEEPSPEED_ROOT" ]]; then
    res=$(deepspeed --help)
    if [ $? -ne 0 ]; then
        echo "please install deepspeed or set DEEPSPEED_ROOT"
    fi
    DEEPSPEED_ROOT=$(pip show deepspeed | grep -i "^Location:" | cut -d" " -f 2)
fi

if [[ -z "$LATEST_CHECKPOINT" ]]; then
    echo "please set HL_LATEST_CHECKPOINT"
    exit 1
else
    LATEST_CHECKPOINT=${LATEST_CHECKPOINT%/}
fi

export PYTHONPATH=$MEGATRON_DEEPSPEED_ROOT:$PYTHONPATH
UNIV_CP_PATH=${LATEST_CHECKPOINT}_universal
mkdir -p $UNIV_CP_PATH
PYTHON_CMD="python ${DEEPSPEED_ROOT}/deepspeed/checkpoint/ds_to_universal.py --input_folder ${LATEST_CHECKPOINT} --output_folder ${UNIV_CP_PATH}"

if [ -n "$EXTRACT_WORKERS" ]; then
    PYTHON_CMD="${PYTHON_CMD} --num_extract_workers ${EXTRACT_WORKERS}"
fi

if [ -n "$MERGE_WORKERS" ]; then
    PYTHON_CMD="${PYTHON_CMD} --num_merge_workers ${MERGE_WORKERS}"
fi

echo $PYTHON_CMD
eval $PYTHON_CMD

if [ $? -ne 0 ]; then
    echo 'Failed to run ds_to_universal.py '
    exit 1
else
    echo "Conversion to universal checkpoint finished. Converted checkpoint available at ${UNIV_CP_PATH} "
    exit 0
fi
