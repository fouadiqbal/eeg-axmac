"""Dataset access to the fold caches produced by Phase A."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import Dataset


class EEGFoldDataset(Dataset):
    """Tensor-backed EEG trials from one subject-disjoint fold split."""

    def __init__(self, cache: Path, split: str = "train", n_classes: int = 4) -> None:
        if split not in ("train", "validation") or n_classes not in (2, 4):
            raise ValueError("split must be train/validation and n_classes must be 2/4")
        payload = torch.load(cache, map_location="cpu", weights_only=True)
        x_key = f"{split}_x2" if n_classes == 2 else f"{split}_x"
        y_key = f"{split}_y2" if n_classes == 2 else f"{split}_y4"
        self.x = payload[x_key]
        self.y = payload[y_key]
        if len(self.x) != len(self.y) or self.x.dtype != torch.float32 or self.y.dtype != torch.int64:
            raise ValueError(f"Invalid cache tensors in {cache}")

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[index], self.y[index]


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a preprocessed EEG fold cache.")
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation"), default="train")
    parser.add_argument("--classes", type=int, choices=(2, 4), default=4)
    args = parser.parse_args()
    dataset = EEGFoldDataset(args.cache, args.split, args.classes)
    print(f"{len(dataset)} windows, X={tuple(dataset.x.shape)}, classes={torch.bincount(dataset.y).tolist()}")


if __name__ == "__main__":
    main()
