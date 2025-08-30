import math
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
from medmnist import INFO
from medmnist import BloodMNIST


@dataclass
class DatasetSplits:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray


def _normalize_images_to_uint8(images: np.ndarray) -> np.ndarray:
    if images.dtype == np.uint8:
        return images
    # images come as [N, H, W, C] in [0,1]
    images = np.clip(images, 0.0, 1.0)
    return (images * 255.0).round().astype(np.uint8)


def load_bloodmnist(val_split: float = 0.1, seed: int = 42) -> DatasetSplits:
    """Load BloodMNIST and return numpy arrays with an explicit val split.

    Returns images as uint8 RGB arrays [N, H, W, C] and labels as int64.
    """
    assert 0.0 <= val_split < 1.0

    info = INFO["bloodmnist"]
    num_classes = len(info["label"])  # noqa: F841 (kept for clarity)

    ds_train = BloodMNIST(split="train", download=True, as_rgb=True)
    ds_val = BloodMNIST(split="val", download=True, as_rgb=True)
    ds_test = BloodMNIST(split="test", download=True, as_rgb=True)

    x_train = _normalize_images_to_uint8(ds_train.imgs)
    y_train = ds_train.labels.reshape(-1).astype(np.int64)

    x_holdout = _normalize_images_to_uint8(ds_val.imgs)
    y_holdout = ds_val.labels.reshape(-1).astype(np.int64)

    # carve validation from train if requested
    if val_split > 0:
        rng = np.random.default_rng(seed)
        num_train = x_train.shape[0]
        num_val = int(math.floor(num_train * val_split))
        indices = rng.permutation(num_train)
        val_idx = indices[:num_val]
        tr_idx = indices[num_val:]
        x_val_from_train = x_train[val_idx]
        y_val_from_train = y_train[val_idx]
        x_train = x_train[tr_idx]
        y_train = y_train[tr_idx]
        # combine with official val split for more stability
        x_val = np.concatenate([x_val_from_train, x_holdout], axis=0)
        y_val = np.concatenate([y_val_from_train, y_holdout], axis=0)
    else:
        x_val = x_holdout
        y_val = y_holdout

    x_test = _normalize_images_to_uint8(ds_test.imgs)
    y_test = ds_test.labels.reshape(-1).astype(np.int64)

    return DatasetSplits(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
    )


