#!/bin/bash

model=${1}  # ./modelzoo/DeepSeek-R1/DeepSeek-R1-Distill-Qwen-7B
device=${2} # 0

ASCEND_RT_VISIBLE_DEVICES=${device} \
python -m real_quantization.real_quantization \
    --model ${model} \
    --method gptq-gptqmodel \
    --w_bits 4 --w_groupsize 128 --w_asym
