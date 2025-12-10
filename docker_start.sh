#!/bin/bash

# Update the vllm-ascend image
export IMAGE=crpi-2k5il0jyr1sa5bw6.cn-hangzhou.personal.cr.aliyuncs.com/fang1767/vllm-ascend-qrmn:latest

# 请根据自己的实际情况和需要修改device对应的npu序号
# 镜像内的Python版本为3.11
docker run --name vllm-ascend02 \
    --device=/dev/davinci_manager \
    --device=/dev/devmm_svm \
    --device=/dev/hisi_hdc \
    --device=/dev/davinci0 \
    --device=/dev/davinci1 \
    --device=/dev/davinci2 \
    --device=/dev/davinci3 \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /root/.cache:/root/.cache \
    -v /data:/data \
    -p 8002:8000 \
    -it -d $IMAGE bash
