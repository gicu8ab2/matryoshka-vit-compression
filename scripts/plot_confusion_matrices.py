import argparse
import glob
import os
import re
import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_dir",
        type=str,
        default="eval_results",
        help="Directory containing *_confusion_matrix.csv files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save confusion matrix plots (default: same as results_dir)",
    )
    return parser.parse_args()


def infer_dataset_from_filename(filename: str) -> str:
    base = os.path.basename(filename)
    parts = base.split("_")
    if len(parts) >= 2:
        return parts[1]
    return ""


def get_class_names(dataset: str, num_classes: int):
    if dataset.lower() == "cifar10" and num_classes == 10:
        return [
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        ]
    if dataset.lower() == "cifar100" and num_classes == 100:
        return [str(i) for i in range(num_classes)]
    return [str(i) for i in range(num_classes)]


def plot_confusion_matrix(cm: np.ndarray, dataset: str, title: str, save_path: str):
    num_classes = cm.shape[0]
    class_names = get_class_names(dataset, num_classes)

    fig, ax = plt.subplots(figsize=(8, 6) if num_classes <= 10 else (14, 12))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(num_classes),
        yticks=np.arange(num_classes),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted label",
        ylabel="True label",
        title=title,
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    if num_classes <= 10:
        thresh = cm.max() / 2.0 if cm.max() > 0 else 0.0
        for i in range(num_classes):
            for j in range(num_classes):
                value = int(cm[i, j])
                if value > 0:
                    ax.text(
                        j,
                        i,
                        str(value),
                        ha="center",
                        va="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontsize=8,
                    )

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    results_dir = args.results_dir
    output_dir = args.output_dir or results_dir

    pattern = os.path.join(results_dir, "*_confusion_matrix.csv")
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"No confusion matrix CSV files found in {results_dir}")
        return

    print(f"Found {len(files)} confusion matrix files in {results_dir}")

    dim_regex = re.compile(r"_dim(\d+)")

    for csv_path in files:
        try:
            cm = np.loadtxt(csv_path, delimiter=",", dtype=np.int64)
        except Exception as e:
            print(f"Skipping {csv_path}: failed to load ({e})")
            continue

        if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
            print(f"Skipping {csv_path}: not a square matrix, shape={cm.shape}")
            continue

        dataset = infer_dataset_from_filename(csv_path)

        base = os.path.basename(csv_path).replace(".csv", "")
        dim_match = dim_regex.search(base)
        dim_str = dim_match.group(1) if dim_match else "?"

        title = f"Confusion Matrix - {dataset} - dim {dim_str}"

        img_name = base + ".png"
        save_path = os.path.join(output_dir, img_name)

        print(f"Plotting {csv_path} -> {save_path}")
        plot_confusion_matrix(cm, dataset, title, save_path)


if __name__ == "__main__":
    main()
