import torch
from torch.utils.data import Dataset
import numpy as np
import random

class LungDataset(Dataset):
    def __init__(self, images_array, masks_array, augment=False):
        """
        Dataset pour la segmentation pulmonaire.
        images_array : np.ndarray de forme (N, 256, 256, 3)
        masks_array  : np.ndarray de forme (N, 256, 256, 1)
        augment : bool, si True applique des augmentations
        """
        self.images = images_array
        self.masks = masks_array
        self.augment = augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].copy()  # (256, 256, 3)
        mask = self.masks[idx].copy()    # (256, 256, 1)

        # ✅ CORRECTION 9: Augmentation de données (optionnel)
        if self.augment and random.random() > 0.5:
            # Rotation horizontale
            if random.random() > 0.5:
                image = np.fliplr(image)
                mask = np.fliplr(mask)
            
            # Rotation verticale
            if random.random() > 0.5:
                image = np.flipud(image)
                mask = np.flipud(mask)
            
            # Rotation 90°
            if random.random() > 0.7:
                k = random.randint(1, 3)
                image = np.rot90(image, k=k, axes=(0, 1))
                mask = np.rot90(mask, k=k, axes=(0, 1))

        # Transposer pour PyTorch (C, H, W)
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        mask = np.transpose(mask, (2, 0, 1)).astype(np.float32)

        # Conversion en tenseur
        image = torch.tensor(image, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32)

        return image, mask

# ✅ CORRECTION 10: Dataset équilibré (optionnel)
class BalancedLungDataset(Dataset):
    def __init__(self, images_array, masks_array, balance_ratio=0.3):
        """
        Dataset équilibré pour la segmentation pulmonaire.
        balance_ratio : proportion minimum de pixels positifs pour inclure une image
        """
        self.images = images_array
        self.masks = masks_array
        
        # Filtrer les images avec assez de pixels positifs
        positive_ratios = masks_array.mean(axis=(1,2,3))
        self.valid_indices = np.where(positive_ratios > balance_ratio)[0]
        
        print(f"Dataset équilibré: {len(self.valid_indices)}/{len(images_array)} images gardées")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        image = self.images[real_idx].copy()
        mask = self.masks[real_idx].copy()

        # Transposer pour PyTorch (C, H, W)
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        mask = np.transpose(mask, (2, 0, 1)).astype(np.float32)

        # Conversion en tenseur
        image = torch.tensor(image, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32)

        return image, mask