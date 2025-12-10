#!/bin/bash

# datasets=("AIME-2025" "AIME-90" "MATH-500" "GSM8K" "GPQA-Diamond" "LiveCodeBench")
datasets=("GPQA-Diamond")
model_path=$1   # ./modelzoo/DeepSeek-R1/DeepSeek-R1-Distill-Qwen-7B
devices=$2      # 0,1,2,3
seed=${3:-42}   # 42 / 43 / 44

for dataset in "${datasets[@]}"; do
    ASCEND_RT_VISIBLE_DEVICES=${devices} \
    python -m inference \
        --model $model_path \
        --dataset $dataset \
        --seed $seed
done
