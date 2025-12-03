# Matryoshka ViT Compression

This repository contains code for Matryoshka Representation Learning (MRL) with Vision Transformers (ViT) on CIFAR-10 and CIFAR-100.

The main training entrypoint is `train_cifar10.py`, which supports:

- **Standard ViT** variants (e.g. `vitfixedposdualpatchnorm`, `vitfixedpos`, `vitlearnedpos`, `vitfixedposvqvae`).
- **Matryoshka Representation Learning (MRL)** variants (`MRL_Entire_Transformer`, `MRL_Entire_Transformer_NestedProj`) with nested embedding dimensions.

Helper scripts under `scripts/` provide ready-made commands for typical experiments (MRL and non-MRL baselines), and additional utilities for evaluation, visualization, and exporting models.

---

## 1. Environment Setup

### 1.1. Python dependencies

The minimal dependencies are listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

This repo assumes a recent PyTorch + torchvision install with CUDA support. For example (adjust versions / CUDA build as needed):

```bash
pip install torch torchvision
```

You will also need common scientific Python packages (if not already installed):

```bash
pip install numpy pandas
```

TensorBoard is already listed in `requirements.txt`.


## 2. Data

Training is implemented for CIFAR-10 and CIFAR-100 via `torchvision.datasets`. By default, datasets are downloaded into `./data` on first run.

No manual pre-processing is required; normalization and augmentations are handled in `train_cifar10.py`.

---

## 3. Basic Training Usage

The core training script is `train_cifar10.py`. It exposes several CLI arguments:

- `--dataset {cifar10,cifar100}`: dataset selection (default: `cifar10`).
- `--net`: model type, e.g.
  - `vitfixedposdualpatchnorm` (default and primary model).
  - `vitfixedpos`, `vitlearnedpos`, `vitfixedposvqvae`.
- `--lr`: learning rate (default `1e-4`).
- `--opt {adam,sgd}`: optimizer choice (default `adam`).
- `--bs`: batch size (default `512`).
- `--size`: input image size (for non-timm ViTs, default `32`).
- `--patch`: patch size for ViT (default `4`).
- `--embedding_dim`: embedding dimension for standard ViT models.
- `--n_epochs`: number of training epochs (default `100`).
- `--dp`: use DataParallel on multiple GPUs.
- `--resume`: resume from a checkpoint in `./checkpoint`.
- `--noaug`: disable RandAugment.
- `--noamp`: disable mixed-precision training.
- `--notb`: disable TensorBoard logging.
- `--mixup`: enable mixup augmentation.

MRL-specific flags:

- `--mrl_flag`: enable Matryoshka Representation Learning.
- `--mrl_dims`: comma-separated embedding dimensions list, e.g. `8,16,32,64,128,256,512`.
- `--nested_proj_mrl`: use the nested-projection MRL variant (`MRL_Entire_Transformer_NestedProj`).

### 3.1. Non-MRL (standard ViT) training

A typical non-MRL run (CIFAR-10, `vitfixedposdualpatchnorm`) looks like:

```bash
python train_cifar10.py \
  --net vitfixedposdualpatchnorm \
  --dataset cifar10 \
  --lr 0.0001 \
  --bs 512 \
  --patch 4 \
  --embedding_dim 128 \
  --n_epochs 500
```

### 3.2. MRL training (nested ViT models)

To train with Matryoshka Representation Learning using the nested-projection variant:

```bash
python train_cifar10.py \
  --nested_proj_mrl \
  --mrl_flag \
  --net vitfixedposdualpatchnorm \
  --dataset cifar10 \
  --lr 0.0001 \
  --opt adam \
  --mrl_dims 8,16,32,64,128,256,512 \
  --bs 512 \
  --patch 4 \
  --n_epochs 500
```


Notes:

- When `--mrl_flag` is set, `train_cifar10.py` uses `Matryoshka_CE_Loss` and expects the model to return a tuple of logits (one per embedding dimension).
- Accuracy during training/validation is computed using the largest (final) nested model.

### 3.3. Using CIFAR-100

To switch to CIFAR-100, change `--dataset`:

```bash
python train_cifar10.py \
  --dataset cifar100 \
  --net vitfixedposdualpatchnorm \
  --lr 0.0001 \
  --bs 512 \
  --patch 4 \
  --embedding_dim 128 \
  --n_epochs 500
```

The script automatically adjusts normalization and number of classes based on the dataset.

---

## 4. Checkpoints, Logs, and TensorBoard

During training, the following artifacts are produced:

- **Checkpoints**: stored under `./checkpoint/` with filenames like
  - `./checkpoint/{net}-{dataset}-{patch}-ckpt.t7`
- **Text logs**: validation metrics per epoch in
  - `./log/log_{net}_{dataset}_patch{patch}.txt`
- **CSV logs**: arrays of validation loss and accuracy in
  - `./log/log_{net}_{dataset}_patch{patch}.csv`
- **TensorBoard logs** (if not disabled with `--notb`): written to
  - `./runs/{net}_lr{lr}_{dataset}`

To visualize training with TensorBoard, run (from the repo root):

```bash
tensorboard --logdir runs
```

Then open the printed URL in a browser.

To resume from a checkpoint, add `--resume` when invoking `train_cifar10.py`. The script will look for `./checkpoint/{net}-{dataset}-{patch}-ckpt.t7`.

---

## 5. Nested model export (MRL only)

If `--mrl_flag` is used, `train_cifar10.py` will, after training, export each nested ViT model to `checkpoint/nested_models/`:

- Files are named like

  ```
  checkpoint/nested_models/{net}_{dataset}_patch{patch}_dim{dim}.pth
  ```

- Each file contains the state dict and metadata (dimension, ViT class, kwargs) so the model can be reconstructed using the helpers in `MRL.py`.
