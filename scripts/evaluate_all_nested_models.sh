#!/bin/bash

DATASET=${1:-cifar10}
NET_TYPE=${2:-vitfixedposdualpatchnorm}
PATCH_SIZE=${3:-4}

MODELS_DIR="checkpoint/nested_models"

echo "==> Evaluating all nested models for ${NET_TYPE} on ${DATASET} with patch size ${PATCH_SIZE}"
echo ""

if [ ! -d "$MODELS_DIR" ]; then
    echo "Error: Models directory not found at $MODELS_DIR"
    echo "Please train a model with --mrl_flag first"
    exit 1
fi

MODEL_PATTERN="${MODELS_DIR}/${NET_TYPE}_${DATASET}_patch${PATCH_SIZE}_dim*.pth"
MODELS=($(ls $MODEL_PATTERN 2>/dev/null))

if [ ${#MODELS[@]} -eq 0 ]; then
    echo "Error: No models found matching pattern: $MODEL_PATTERN"
    exit 1
fi

echo "Found ${#MODELS[@]} models to evaluate:"
for model in "${MODELS[@]}"; do
    echo "  - $(basename $model)"
done
echo ""

for model in "${MODELS[@]}"; do
    echo "=========================================="
    echo "Evaluating: $(basename $model)"
    echo "=========================================="
    python separate_vit_evaluator.py \
        --model_path "$model" \
        --dataset "$DATASET" \
        --bs 100 \
        --size 32
    echo ""
done

echo "Results saved to eval_results/"
echo "TensorBoard logs saved to runs/eval_*"
