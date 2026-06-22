#!/usr/bin/env python3
"""Generate MONZA MNIST client partitions.

The MONZA runtime reads ../dataset/MNIST/{train,test}/{client_id}.npz with a
single object named "data" containing {"x": ndarray, "y": ndarray}.
"""
from __future__ import annotations

import random
import shutil
import argparse
from pathlib import Path

import numpy as np
from torchvision.datasets import MNIST


NUM_CLIENTS = 100
NUM_CLASSES = 10
DIRICHLET_ALPHA = 0.1
TRAIN_RATIO = 0.75
PUBLIC_VAL_RATIO = 0.05
MIN_CLIENT_SAMPLES = 40
SEED = 1


def load_mnist(root: Path) -> tuple[np.ndarray, np.ndarray]:
    train = MNIST(root=str(root), train=True, download=True)
    test = MNIST(root=str(root), train=False, download=True)
    x = np.concatenate(
        [
            train.data.numpy()[:, None, :, :],
            test.data.numpy()[:, None, :, :],
        ],
        axis=0,
    ).astype(np.float32)
    y = np.concatenate([train.targets.numpy(), test.targets.numpy()], axis=0).astype(np.int64)
    x /= 255.0
    return x, y


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--num-clients", type=int, default=NUM_CLIENTS)
    parser.add_argument("--seed", type=int, default=SEED)
    args, _unknown = parser.parse_known_args()
    if args.num_clients <= 0:
        raise SystemExit("--num-clients deve ser positivo.")
    return args


def dirichlet_partition(y: np.ndarray, rng: np.random.Generator, num_clients: int) -> list[np.ndarray]:
    class_indices = [np.where(y == c)[0] for c in range(NUM_CLASSES)]
    for indices in class_indices:
        rng.shuffle(indices)

    while True:
        client_indices = [[] for _ in range(num_clients)]
        for indices in class_indices:
            proportions = rng.dirichlet(np.repeat(DIRICHLET_ALPHA, num_clients))
            split_points = (np.cumsum(proportions)[:-1] * len(indices)).astype(int)
            for client_id, split in enumerate(np.split(indices, split_points)):
                client_indices[client_id].extend(split.tolist())

        sizes = [len(indices) for indices in client_indices]
        if min(sizes) >= MIN_CLIENT_SAMPLES:
            break
        print(
            f"Client data size does not meet the minimum requirement {MIN_CLIENT_SAMPLES}. "
            "Try allocating again."
        )

    out = []
    for indices in client_indices:
        arr = np.asarray(indices, dtype=np.int64)
        rng.shuffle(arr)
        out.append(arr)
    return out


def save_client_split(out_dir: Path, client_id: int, x: np.ndarray, y: np.ndarray) -> tuple[int, int, int]:
    train_end = int(len(x) * (TRAIN_RATIO - PUBLIC_VAL_RATIO))
    public_val_end = int(len(x) * TRAIN_RATIO)
    train_x, public_val_x, test_x = x[:train_end], x[train_end:public_val_end], x[public_val_end:]
    train_y, public_val_y, test_y = y[:train_end], y[train_end:public_val_end], y[public_val_end:]
    np.savez_compressed(out_dir / "train" / f"{client_id}.npz", data={"x": train_x, "y": train_y})
    np.savez_compressed(out_dir / "public_val" / f"{client_id}.npz", data={"x": public_val_x, "y": public_val_y})
    np.savez_compressed(out_dir / "test" / f"{client_id}.npz", data={"x": test_x, "y": test_y})
    return len(train_y), len(public_val_y), len(test_y)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    base_dir = Path(__file__).resolve().parent
    out_dir = base_dir / "MNIST"
    raw_dir = base_dir / "raw"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "train").mkdir(parents=True)
    (out_dir / "public_val").mkdir(parents=True)
    (out_dir / "test").mkdir(parents=True)

    x_all, y_all = load_mnist(raw_dir)
    partitions = dirichlet_partition(y_all, rng, args.num_clients)

    train_counts = []
    public_val_counts = []
    test_counts = []
    for client_id, indices in enumerate(partitions):
        x_client = x_all[indices]
        y_client = y_all[indices]
        labels, counts = np.unique(y_client, return_counts=True)
        print(f"Client {client_id}\t Size of data: {len(indices)}\t Labels:  {labels}")
        print(f"\t\t Samples of labels:  {list(zip(labels.tolist(), counts.tolist()))}")
        print("-" * 50)
        n_train, n_public_val, n_test = save_client_split(out_dir, client_id, x_client, y_client)
        train_counts.append(n_train)
        public_val_counts.append(n_public_val)
        test_counts.append(n_test)

    print(f"Total number of samples: {len(y_all)}")
    print(f"The number of train samples: {train_counts}")
    print(f"The number of public_val samples: {public_val_counts}")
    print(f"The number of test samples: {test_counts}")
    print("\nSaving to disk.\n")
    print("Finish generating dataset.")


if __name__ == "__main__":
    main()
