#! /bin/bash


# python train_cifar10.py \
#     --mrl_flag \
#     --dataset 'cifar100' \
#     --lr '0.0001' \
#     --opt 'adam' \
#     --mrl_dims '8,16,32,64,128,256,512' \
#     --net vitfixedposdualpatchnorm \
#     --bs 512 \
#     --patch 4 \
#     --n_epochs 500

python train_cifar10.py \
    --nested_proj_mrl \
    --mrl_flag \
    --net vitfixedposdualpatchnorm \
    --dataset 'cifar10' \
    --lr '0.0001' \
    --opt 'adam' \
    --mrl_dims '8,16,32,64,128,256,512' \
    --bs 512 \
    --patch 4 \
    --n_epochs 500
