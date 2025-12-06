#!/usr/bin/env python

from __future__ import annotations
import argparse
import os
import re
from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from models.vit import ViT_LearnPos, ViTFixedPos, ViTFixedPosDualPatchNorm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="t-SNE on nested ViT embeddings (Matryoshka models)")
    parser.add_argument("--nested_dir", type=str, default="checkpoint/nested_models",
                        help="Directory containing nested model checkpoints")
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100"],
                        help="Dataset used for training the checkpoints")
    parser.add_argument("--bs", type=int, default=256, help="Batch size for feature extraction")
    parser.add_argument("--size", type=int, default=32, help="Image resize size")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of data loading workers")
    parser.add_argument("--max_samples", type=int, default=5000,
                        help="Maximum number of test samples to use for t-SNE (per model). Use -1 for all.")
    parser.add_argument("--perplexity", type=float, default=30.0, help="t-SNE perplexity")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed for t-SNE")
    parser.add_argument("--output_dir", type=str, default="tsne_plots",
                        help="Directory to save t-SNE PNGs")
    return parser.parse_args()


def get_dataset_and_loader(args: argparse.Namespace):
    size = args.size
    if args.dataset == "cifar10":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        dataset_class = torchvision.datasets.CIFAR10
    else:  # cifar100
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
        dataset_class = torchvision.datasets.CIFAR100
    transform_test = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    testset = dataset_class(root="./data", train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=args.bs,
        shuffle=False,
        num_workers=args.num_workers,
    )
    return testset, testloader


def discover_checkpoints(nested_dir: str, dataset: str) -> List[Tuple[str, int]]:
    if not os.path.isdir(nested_dir):
        raise FileNotFoundError(f"Nested checkpoint directory not found: {nested_dir}")
    pattern = re.compile(rf"vitfixedposdualpatchnorm_{dataset}_patch\d+_dim(\d+)\.pth")
    ckpts: List[Tuple[str, int]] = []
    for fname in os.listdir(nested_dir):
        m = pattern.match(fname)
        if m:
            dim = int(m.group(1))
            ckpts.append((os.path.join(nested_dir, fname), dim))
    ckpts.sort(key=lambda x: x[1])
    if not ckpts:
        raise RuntimeError(f"No nested checkpoints found in {nested_dir} for dataset '{dataset}'")
    return ckpts


def load_vit_checkpoint(path: str, device: torch.device) -> Tuple[nn.Module, int, str]:
    print(f"\n==> Loading checkpoint: {path}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=device)
    vit_class_name = checkpoint["vit_class"]
    if vit_class_name == "ViTFixedPosDualPatchNorm":
        vit_class = ViTFixedPosDualPatchNorm
    elif vit_class_name == "ViTFixedPos":
        vit_class = ViTFixedPos
    elif vit_class_name == "ViT_LearnPos":
        vit_class = ViT_LearnPos
    else:
        raise ValueError(f"Unknown ViT class in checkpoint: {vit_class_name}")
    vit_kwargs = checkpoint["vit_kwargs"].copy()
    vit_kwargs["dim"] = checkpoint["dim"]
    net = vit_class(**vit_kwargs)
    net.load_state_dict(checkpoint["model_state_dict"])
    net = net.to(device)
    if device.type == "cuda":
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    dim = checkpoint["dim"]
    print(f"Loaded {vit_class_name} with dim={dim}")
    return net, dim, vit_class_name


def collect_embeddings(
    net: nn.Module,
    loader,
    device: torch.device,
    max_samples: int,
) -> Tuple[np.ndarray, np.ndarray]:
    net.eval()

    if isinstance(net, nn.DataParallel):
        module = net.module
    else:
        module = net
    embeddings: List[torch.Tensor] = []
    labels: List[torch.Tensor] = []
    def hook(_module, _inputs, output):
        embeddings.append(output.detach().cpu())
    handle = module.to_latent.register_forward_hook(hook)
    with torch.no_grad():
        seen = 0
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            if max_samples > 0 and seen >= max_samples:
                break
            if max_samples > 0 and seen + inputs.size(0) > max_samples:
                keep = max_samples - seen
                inputs = inputs[:keep]
                targets = targets[:keep]
            outputs = net(inputs)  # hook captures latent embeddings
            labels.append(targets.detach().cpu())
            seen += inputs.size(0)
    handle.remove()
    if not embeddings:
        raise RuntimeError("No embeddings were collected. Check the hook and model forward.")
    emb_arr = torch.cat(embeddings, dim=0).numpy()
    label_arr = torch.cat(labels, dim=0).numpy()
    return emb_arr, label_arr

def run_tsne_and_plot(
    embeddings: np.ndarray,
    labels: np.ndarray,
    dim: int,
    dataset: str,
    args: argparse.Namespace,
):
    print(f"Running t-SNE for dim={dim} on {embeddings.shape[0]} samples...")
    tsne = TSNE(
        n_components=2,
        perplexity=args.perplexity,
        random_state=args.random_state,
        init="pca",
        learning_rate="auto",
    )
    emb_2d = tsne.fit_transform(embeddings)
    os.makedirs(args.output_dir, exist_ok=True)
    fname = os.path.join(args.output_dir, f"tsne_{dataset}_dim{dim}.png")
    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=labels, cmap="tab10", s=5, alpha=0.7)
    plt.title(
        f"t-SNE of ViT embeddings\n(dataset={dataset}, dim={dim})",
        fontsize=30,
    )
    plt.xlabel("t-SNE-1", fontsize=25)
    plt.ylabel("t-SNE-2", fontsize=25)
    cbar = plt.colorbar(scatter)
    cbar.set_label("Class index", fontsize=25)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    cbar.ax.tick_params(labelsize=12)
    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    plt.close()
    print(f"Saved t-SNE plot to {fname}")

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("==> Preparing data..")
    testset, testloader = get_dataset_and_loader(args)
    print(f"Test samples: {len(testset)}")
    print("==> Discovering nested checkpoints..")
    checkpoints = discover_checkpoints(args.nested_dir, args.dataset)
    print("Found checkpoints:")
    for path, dim in checkpoints:
        print(f"  dim={dim}: {os.path.basename(path)}")
    for path, dim in checkpoints:
        net, ckpt_dim, vit_class_name = load_vit_checkpoint(path, device)
        assert ckpt_dim == dim, "Checkpoint dim mismatch"
        embeddings, labels = collect_embeddings(net, testloader, device, args.max_samples)
        run_tsne_and_plot(embeddings, labels, dim, args.dataset, args)
    print("\nAll done. t-SNE plots saved to:", args.output_dir)

if __name__ == "__main__":
    main()
