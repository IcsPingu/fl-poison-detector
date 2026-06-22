#!/usr/bin/env python3
"""Create MONZA train_mal/ files for reverse label-flip attacks.

The MONZA dataset generator writes per-client train/test .npz files. The
malicious label-flip client loader expects a parallel train_mal/ directory with
the same client files but inverted labels: y -> num_classes - 1 - y.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("PFLlibMonza/dataset/MNIST"),
        help="Dataset directory containing train/*.npz.",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=10,
        help="Number of classes used by the reverse flip.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_dir = args.dataset_dir / "train"
    train_mal_dir = args.dataset_dir / "train_mal"
    if not train_dir.exists():
        raise SystemExit(f"Missing train directory: {train_dir}")

    train_mal_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    total_samples = 0
    for src in sorted(train_dir.glob("*.npz"), key=lambda p: int(p.stem)):
        with np.load(src, allow_pickle=True) as npz:
            data = npz["data"].tolist()
        x = data["x"]
        y = np.asarray(data["y"], dtype=np.int64)
        flipped = (args.num_classes - 1 - y).astype(np.int64)
        out_data = {"x": x, "y": flipped}
        np.savez_compressed(train_mal_dir / src.name, data=out_data)
        count += 1
        total_samples += int(flipped.shape[0])

    print(
        f"Created {count} train_mal files in {train_mal_dir} "
        f"with {total_samples} flipped samples."
    )


if __name__ == "__main__":
    main()
