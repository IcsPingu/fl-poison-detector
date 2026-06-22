#!/usr/bin/env python3
"""Create label-flipped MONZA train_mal files from an existing train split."""
from __future__ import annotations

import argparse
from pathlib import Path


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
        help="Number of classes. Flip mapping is y_flip = num_classes - 1 - y.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files in train_mal/.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import numpy as np

    train_dir = args.dataset_dir / "train"
    out_dir = args.dataset_dir / "train_mal"
    if not train_dir.exists():
        raise FileNotFoundError(f"train/ nao encontrado: {train_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0
    for src in sorted(train_dir.glob("*.npz"), key=lambda p: int(p.stem)):
        dst = out_dir / src.name
        if dst.exists() and not args.overwrite:
            skipped += 1
            continue
        with np.load(src, allow_pickle=True) as loaded:
            data = loaded["data"].tolist()
        y = np.asarray(data["y"], dtype=np.int64)
        data["y"] = (args.num_classes - 1 - y).astype(np.int64)
        np.savez_compressed(dst, data=data)
        written += 1

    print(f"train_mal criado em {out_dir}")
    print(f"arquivos escritos={written} pulados={skipped}")


if __name__ == "__main__":
    main()
