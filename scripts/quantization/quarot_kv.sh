#!/bin/bash

model=${1}  # ./modelzoo/DeepSeek-R1/DeepSeek-R1-Distill-Qwen-7B
tp=${2}     # 4
device=${3} # 0

model_name=$(basename "$model")

bits=("3" "4")
for BITS in "${bits[@]}"; do
    ASCEND_RT_VISIBLE_DEVICES=${device} \
    python -m methods.quarot_gptq.save_fake_quant \
        --model ${model} \
        --k_bits ${BITS} --k_asym --k_groupsize 128 --k_clip_ratio 0.95 \
        --v_bits ${BITS} --v_asym --v_groupsize 128 --v_clip_ratio 0.95 \
        --tp ${tp} \
        --save_qmodel_path ./outputs/modelzoo/quarot/${model_name}-quarot-kv${BITS}-tp${tp}
done
