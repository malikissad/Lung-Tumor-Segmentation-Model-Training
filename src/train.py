import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import sys
import os
sys.path.append(os.path.abspath("Swin-Unet"))

from networks.vision_transformer import SwinUnet
from config import get_config
import types

from src.datasets import LungDataset

# Fonction Dice
def dice_score(preds, targets, threshold=0.5):
    preds = (preds > threshold).float()
    intersection = (preds * targets).sum(dim=(1,2,3))
    union = preds.sum(dim=(1,2,3)) + targets.sum(dim=(1,2,3))
    dice = (2. * intersection + 1e-8) / (union + 1e-8)
    return dice.mean().item()

# Fonction Accuracy
def pixel_accuracy(preds, targets, threshold=0.5):
    preds = (preds > threshold).float()
    correct = (preds == targets).float()
    return correct.mean().item()

# Entraînement + Évaluation (tout en 1)
def Train(image_path, mask_path, image_path_test, mask_path_test, num_epochs=20):
    # 1. Charger les données
    X_train = np.load(image_path)
    Y_train = np.load(mask_path)
    X_test = np.load(image_path_test)
    Y_test = np.load(mask_path_test)

    train_dataset = LungDataset(X_train, Y_train)
    test_dataset = LungDataset(X_test, Y_test)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # 2. Config Swin-Unet
    args = types.SimpleNamespace()
    args.cfg = "Swin-Unet/configs/swin_tiny_patch4_window7_224_lite.yaml"
    args.opts = None
    args.batch_size = None
    args.zip = None
    args.cache_mode = None
    args.resume = None
    args.accumulation_steps = None
    args.use_checkpoint = False
    args.amp_opt_level = None
    args.tag = None
    args.eval = False
    args.throughput = False

    config = get_config(args)
    config.defrost()
    config.DATA.IMG_SIZE = 256
    config.MODEL.SWIN.IN_CHANS = 3
    config.MODEL.NUM_CLASSES = 1
    config.freeze()

    # 3. Modèle
    model = SwinUnet(config=config, img_size=256, num_classes=1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(3.0).to(device))
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # 4. Boucle d'entraînement
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)

        #Test + Accuracy
        model.eval()
        dice_total = 0.0
        acc_total = 0.0
        with torch.no_grad():
            for images, masks in test_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                dice_total += dice_score(outputs, masks)
                acc_total += pixel_accuracy(outputs, masks)
        dice_avg = dice_total / len(test_loader)
        acc_avg = acc_total / len(test_loader)

        print(f"Epoch [{epoch+1}/{num_epochs}] — Loss: {avg_loss:.4f} | Dice: {dice_avg:.4f} | Acc: {acc_avg:.4f}")

    # 5. Sauvegarde
    torch.save(model.state_dict(), "swin_unet_lung.pth")
    print("✅ Modèle sauvegardé sous 'swin_unet_lung.pth'")
