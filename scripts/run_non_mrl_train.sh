#! /bin/bash


# python train_cifar10.py \
#     --net vitfixedposvqvae \
#     --patch 4 \
#     --embedding_dim 8 \
#     --n_epochs 500

python train_cifar10.py \
    --net vitfixedposdualpatchnorm \
    --lr '0.0001' \
    --bs 512 \
    --patch 4 \
    --embedding_dim 128 \
    --n_epochs 500