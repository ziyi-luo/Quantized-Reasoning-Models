#!/bin/bash

datasets=("ARC-E" "ARC-C" "HellaSwag" "LAMBADA" "PIQA" "WinoGrande")
model_path=$1   # ./modelzoo/Meta-Llama-3-8B
devices=$2      # 0,1,2,3
seed=${3:-42}   # 42 / 43 / 44

for dataset in "${datasets[@]}"; do
    ASCEND_RT_VISIBLE_DEVICES=${devices} \
    python -m inference_qa \
        --model $model_path \
        --dataset $dataset \
        --seed $seed
done
