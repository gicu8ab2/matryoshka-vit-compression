#!/bin/bash

docker run \
  -it \
  --name vit-mrl \
  --rm \
  --privileged \
  --gpus=all \
  --shm-size=32g \
  --network host \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e NCCL_DEBUG=INFO \
  -e NCCL_IB_DISABLE=0 \
  -e NCCL_NET_GDR_LEVEL=2 \
  -e NCCL_TREE_THRESHOLD=0 \
  -v /home/ubuntu/robert.taylor:/home/ubuntu/robert.taylor \
  deimv2:gh200 \
  /bin/bash

