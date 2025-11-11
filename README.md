# Matryoshka Vision Transformer Compression

The goal of this project is use Matryoshka representation learning with respect to the embedding dimension to compress vision transformer models.  I 
nest the embedding dimension from small to large exponentially and keep the nested embedding dimensions fixed end-to-end through the entire
transformer architecture from patch embedding outputs to the final layer of the final transformer block.

## Acknowledgement
This code is forked from https://github.com/kentaroy47/vision-transformers-cifar10

```
@misc{yoshioka2024visiontransformers,
  author       = {Kentaro Yoshioka},
  title        = {vision-transformers-cifar10: Training Vision Transformers (ViT) and related models on CIFAR-10},
  year         = {2024},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/kentaroy47/vision-transformers-cifar10}}
}
```
and borrows functionality from https://github.com/RAIVNLab/MRL



## Overview

When training with `--mrl_flag`, the `MRL_Entire_Transformer` creates multiple ViT models with different embedding dimensions (e.g., 32, 128, 256, 512). After training, each nested model is automatically saved separately for independent evaluation and deployment.

## Training with MRL

Train a model with nested representations:

```bash
python train_cifar10.py \
    --net vitfixedposdualpatchnorm \
    --mrl_flag \
    --mrl_dims 32,128,256,512 \
    --dataset cifar10 \
    --n_epochs 100
```

After training completes, individual models will be saved to:
```
checkpoint/nested_models/
├── vitfixedposdualpatchnorm_cifar10_patch4_dim32.pth
├── vitfixedposdualpatchnorm_cifar10_patch4_dim128.pth
├── vitfixedposdualpatchnorm_cifar10_patch4_dim256.pth
└── vitfixedposdualpatchnorm_cifar10_patch4_dim512.pth
```

## Evaluating Individual Models

### Option 1: Evaluate a Single Model

```bash
python separate_vit_evaluator.py \
    --model_path checkpoint/nested_models/vitfixedposdualpatchnorm_cifar10_patch4_dim512.pth \
    --dataset cifar10 \
    --bs 100
```

### Option 2: Evaluate All Nested Models

```bash
bash evaluate_all_nested_models.sh cifar10 vitfixedposdualpatchnorm 4
```

Or simply:
```bash
bash evaluate_all_nested_models.sh
```
(uses default values: cifar10, vitfixedposdualpatchnorm, patch size 4)

## Output

### Console Output
- Real-time progress bar
- Overall accuracy and loss
- Per-class accuracy breakdown
- Evaluation time

### Files Generated

1. **TensorBoard Logs**: `runs/eval_<model_name>/`
   - Evaluation/Loss
   - Evaluation/Accuracy
   - Evaluation/Time
   - PerClass/Class_0, Class_1, etc.
   - Model_Info (text summary)

2. **Text Results**: `eval_results/<model_name>_results.txt`
   - Complete evaluation summary
   - Model configuration
   - Per-class accuracy

## Viewing Results in TensorBoard

```bash
tensorboard --logdir=runs
```

Then open http://localhost:6006 in your browser.

You can compare all nested models side-by-side to see the accuracy/efficiency trade-offs.

## Using Models for Inference

### Load a saved model in Python:

```python
import torch
from models.vit import ViTFixedPosDualPatchNorm

# Load checkpoint
checkpoint = torch.load('checkpoint/nested_models/vitfixedposdualpatchnorm_cifar10_patch4_dim512.pth')

# Recreate model
vit_kwargs = checkpoint['vit_kwargs'].copy()
vit_kwargs['dim'] = checkpoint['dim']
model = ViTFixedPosDualPatchNorm(**vit_kwargs)

# Load weights
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Run inference
with torch.no_grad():
    output = model(input_tensor)
```

## Model Size Comparison

The nested models provide different accuracy/efficiency trade-offs:

- **dim=32**: Smallest, fastest, lowest accuracy
- **dim=128**: Small, fast, moderate accuracy
- **dim=256**: Medium size, good accuracy
- **dim=512**: Largest, best accuracy, slower

Use smaller models for:
- Edge devices with limited compute
- Real-time applications
- Batch processing with tight latency requirements

Use larger models for:
- Server-side inference
- Applications requiring maximum accuracy
- When compute resources are available

## Advanced Usage

### Custom Evaluation Script

You can modify `separate_vit_evaluator.py` to:
- Add custom metrics
- Test on different datasets
- Perform error analysis
- Generate confusion matrices
- Measure inference speed

### Batch Evaluation with Different Settings

```bash
# Evaluate with different batch sizes
for bs in 50 100 200; do
    python separate_vit_evaluator.py \
        --model_path checkpoint/nested_models/vitfixedposdualpatchnorm_cifar10_patch4_dim512.pth \
        --dataset cifar10 \
        --bs $bs
done
```

## Troubleshooting

**Error: Model checkpoint not found**
- Ensure you've trained with `--mrl_flag`
- Check that training completed successfully
- Verify the path to the checkpoint

**Error: CUDA out of memory**
- Reduce batch size with `--bs 50` or lower
- Evaluate on CPU (remove CUDA device)

**Error: Unknown ViT class**
- Ensure the model was trained with a supported ViT variant
- Check that the imports in `separate_vit_evaluator.py` match your model

## Example Workflow

```bash
# 1. Train with MRL
python train_cifar10.py --net vitfixedposdualpatchnorm --mrl_flag --mrl_dims 32,128,256,512

# 2. Evaluate all nested models
bash evaluate_all_nested_models.sh

# 3. View results in TensorBoard
tensorboard --logdir=runs

# 4. Compare model sizes and accuracies
ls -lh checkpoint/nested_models/

# 5. Deploy the best model for your use case
# (e.g., dim=128 for edge devices, dim=512 for servers)
```
