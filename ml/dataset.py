"""
PyTorch Dataset and DataLoader for Bi-Temporal Change Detection.
Compatible with standard LEVIR-CD, OSCD, and WHU-CD dataset folder hierarchies:
  root/
    ├── A/       (T-0 Reference Images)
    ├── B/       (T-1 Target Images)
    └── label/   (Ground Truth Binary Change Masks)
"""

import os
import glob
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

class SatellitePairDataset(Dataset):
    """
    Dataset loader for paired temporal satellite images and change ground truth.
    """
    def __init__(
        self,
        root_dir: str = None,
        file_pairs: list[dict] = None,
        image_size: tuple[int, int] = (256, 256),
        is_train: bool = True,
    ):
        """
        Args:
            root_dir: Directory containing 'A', 'B', and optionally 'label' subdirectories.
            file_pairs: Direct list of dicts [{'path_t0': ..., 'path_t1': ..., 'path_label': ...}].
            image_size: Spatial dimensions to resize or crop to.
            is_train: If True, applies random horizontal/vertical flip augmentations.
        """
        self.image_size = image_size
        self.is_train = is_train
        self.samples = []

        if file_pairs:
            self.samples = file_pairs
        elif root_dir and os.path.isdir(root_dir):
            dir_a = os.path.join(root_dir, "A")
            dir_b = os.path.join(root_dir, "B")
            dir_label = os.path.join(root_dir, "label")

            files_a = sorted(glob.glob(os.path.join(dir_a, "*.*")))
            for path_a in files_a:
                filename = os.path.basename(path_a)
                path_b = os.path.join(dir_b, filename)
                path_lbl = os.path.join(dir_label, filename) if os.path.isdir(dir_label) else None

                if os.path.exists(path_b):
                    self.samples.append({
                        "path_t0": path_a,
                        "path_t1": path_b,
                        "path_label": path_lbl if (path_lbl and os.path.exists(path_lbl)) else None,
                    })

    def __len__(self) -> int:
        return len(self.samples)

    def _normalize_image(self, img_pil: Image.Image) -> torch.Tensor:
        """Converts PIL RGB image to normalized FloatTensor with ImageNet statistics."""
        img = img_pil.resize(self.image_size, Image.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0
        if len(arr.shape) == 2:
            arr = np.stack([arr] * 3, axis=-1)

        # Standard ImageNet Mean/Std
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        norm_arr = (arr - mean) / std

        # Transpose to (C, H, W)
        tensor = torch.from_numpy(norm_arr.transpose(2, 0, 1)).float()
        return tensor

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        img_t0 = Image.open(sample["path_t0"]).convert("RGB")
        img_t1 = Image.open(sample["path_t1"]).convert("RGB")

        lbl_tensor = None
        if sample.get("path_label"):
            lbl_pil = Image.open(sample["path_label"]).convert("L")
            lbl_pil = lbl_pil.resize(self.image_size, Image.NEAREST)
            lbl_arr = (np.array(lbl_pil, dtype=np.float32) > 127).astype(np.float32)
            lbl_tensor = torch.from_numpy(lbl_arr).unsqueeze(0).float()
        else:
            # Dummy label if evaluating without ground truth
            lbl_tensor = torch.zeros((1, self.image_size[1], self.image_size[0]), dtype=torch.float32)

        # Apply synchronized data augmentations during training
        if self.is_train:
            # Random Horizontal Flip
            if random.random() > 0.5:
                img_t0 = img_t0.transpose(Image.FLIP_LEFT_RIGHT)
                img_t1 = img_t1.transpose(Image.FLIP_LEFT_RIGHT)
                lbl_tensor = torch.flip(lbl_tensor, dims=[2])

            # Random Vertical Flip
            if random.random() > 0.5:
                img_t0 = img_t0.transpose(Image.FLIP_TOP_BOTTOM)
                img_t1 = img_t1.transpose(Image.FLIP_TOP_BOTTOM)
                lbl_tensor = torch.flip(lbl_tensor, dims=[1])

        t0_tensor = self._normalize_image(img_t0)
        t1_tensor = self._normalize_image(img_t1)

        return {
            "img_t0": t0_tensor,
            "img_t1": t1_tensor,
            "label": lbl_tensor,
            "path_t0": sample["path_t0"],
            "path_t1": sample["path_t1"],
        }
