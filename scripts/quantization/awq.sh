#!/bin/bash

model=${1}  # ./modelzoo/DeepSeek-R1/DeepSeek-R1-Distill-Qwen-7B
tp=${2}     # 4
device=${3} # 0

model_name=$(basename "$model")

bits=("3")
for BITS in "${bits[@]}"; do
    ASCEND_VISIBLE_DEVICES=${device} \
    python -m methods.awq.run_awq \
        --model ${model} \
        --w_bits ${BITS} --w_groupsize 128 --w_asym \
        --save_qmodel_path ./outputs/modelzoo/awq/${model_name}-awq-w${BITS}g128-tp${tp}
done
