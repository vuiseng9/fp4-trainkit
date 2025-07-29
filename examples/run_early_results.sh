#!/usr/bin/env bash

# Script to run FP32 baseline and FP4 training with different recipes
# on a very small ViTOneBlock on MNIST dataset
# for early results and testing purposes

datetimestr() {
    date +"%m%d_%H%M"
}

NEPOCHS=5

mkdir -p logs

# Baseline FP32 training
echo "[Info] Running FP32 baseline training..."
python fp32_train_ViTOneBlock_mnist.py --epochs $NEPOCHS 2>&1 | tee logs/fp32_$(datetimestr).log

# Run MXFP4 with different recipes
recipes=("mx_baseline" "nvidia_round_to_infinity" "tetrajet")
for recipe in "${recipes[@]}"; do
    echo "[Info] Running MXFP4 training with recipe: $recipe"
    python fp4_train_ViTOneBlock_mnist.py --recipe $recipe --epochs $NEPOCHS 2>&1 | tee logs/mxfp4_${recipe}_$(datetimestr).log
done

# Run NVFP4 need a different learning rate
python fp4_train_ViTOneBlock_mnist.py --recipe fp4_all_the_way --lr 5e-4 --epochs $NEPOCHS 2>&1 | tee logs/nvfp4_all_the_way_$(datetimestr).log