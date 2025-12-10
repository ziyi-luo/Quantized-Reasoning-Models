#!/bin/bash

model=${1}  # ./modelzoo/DeepSeek-R1/DeepSeek-R1-Distill-Qwen-7B
tp=${2}     # 4
device=${3} # 0

model_name=$(basename "$model")

bits=("4" "8")
for BITS in "${bits[@]}"; do
    ASCEND_RT_VISIBLE_DEVICES=${device} \
    python -m methods.quarot_gptq.save_fake_quant \
        --model ${model} \
        --rotate \
        --w_bits ${BITS} --w_clip \
        --a_bits ${BITS} --a_clip_ratio 0.9 --a_asym \
        --k_bits ${BITS} --k_asym --k_groupsize 128 --k_clip_ratio 0.95 \
        --v_bits ${BITS} --v_asym --v_groupsize 128 --v_clip_ratio 0.95 \
        --tp ${tp} \
        --save_qmodel_path ./outputs/modelzoo/quarot/${model_name}-quarot-w${BITS}a${BITS}kv${BITS}-tp${tp} \
        --cal_dataset reasoning-numina-math-1.5
done
