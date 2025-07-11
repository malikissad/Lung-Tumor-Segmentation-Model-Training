import torch
from torch.utils.data import Dataset
import numpy as np

class LungDataset(Dataset):
    def __init__(self, images_array, masks_array):
        """
        Dataset pour la segmentation pulmonaire.
        images_array : np.ndarray de forme (N, 256, 256, 3)
        masks_array  : np.ndarray de forme (N, 256, 256, 1)
        """
        self.images = images_array
        self.masks = masks_array

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]  # (256, 256, 3)
        mask = self.masks[idx]    # (256, 256, 1)

        # Transposer pour PyTorch (C, H, W)
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        mask = np.transpose(mask, (2, 0, 1)).astype(np.float32)

        # Conversion en tenseur
        image = torch.tensor(image, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32)

        return image, mask
