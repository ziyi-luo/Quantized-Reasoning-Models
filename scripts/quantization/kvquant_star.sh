#!/bin/bash

model=${1}  # ./modelzoo/DeepSeek-R1/DeepSeek-R1-Distill-Qwen-7B
tp=${2}     # 4
device=${3} # 0,1,2,3

model_name=$(basename "$model")

bits=("3" "4")
for BITS in "${bits[@]}"; do
    if [[ "$model_name" == "DeepSeek-R1-Distill-Qwen-1.5B" ]] || [[ "$model_name" == "DeepSeek-R1-Distill-Qwen-7B" ]]; then
        ASCEND_RT_VISIBLE_DEVICES=${device} \
        python -m methods.kvquant_star.save_fake_quant \
            --model ${model} \
            --k_bits ${BITS} --k_asym --k_pre_bias \
            --v_bits ${BITS} --v_asym --v_groupsize 128 \
            --seqlen 512 --nsamples 512 --cal_dataset pileval \
            --tp ${tp} \
            --save_qmodel_path ./outputs/modelzoo/kvquant_star/${model_name}-kvquant_star-kv${BITS}-tp${tp}
    else
        ASCEND_RT_VISIBLE_DEVICES=${device} \
        python -m methods.kvquant_star.save_fake_quant \
            --model ${model} \
            --k_bits ${BITS} --k_asym \
            --v_bits ${BITS} --v_asym --v_groupsize 128 \
            --seqlen 512 --nsamples 512 --cal_dataset pileval \
            --tp ${tp} \
            --save_qmodel_path ./outputs/modelzoo/kvquant_star/${model_name}-kvquant_star-kv${BITS}-tp${tp}
    fi
done
