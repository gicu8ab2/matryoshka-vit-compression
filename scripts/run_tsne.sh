#! /bin/bash

python tsne_nested_vit_embeddings.py \
  --nested_dir checkpoint/nested_models \
  --dataset cifar10 \
  --bs 256 \
  --max_samples 5000 \
  --perplexity 30.0 \
  --output_dir tsne_plots