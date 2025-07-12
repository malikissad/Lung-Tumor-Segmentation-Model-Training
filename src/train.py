import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

import sys
import os
sys.path.append(os.path.abspath("Swin-Unet"))

from networks.vision_transformer import SwinUnet
from config import get_config
import types

from src.datasets import LungDataset

# Dice + BCE Loss
class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True, pos_weight=None):
        super(DiceBCELoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
    def forward(self, inputs, targets, smooth=1):
        bce_loss = self.bce(inputs, targets)
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        dice_loss = 1 - dice
        return bce_loss + dice_loss

# Dice
def dice_score(preds, targets, threshold=0.5):
    preds = torch.sigmoid(preds)
    preds = (preds > threshold).float()
    intersection = (preds * targets).sum(dim=(1,2,3))
    union = preds.sum(dim=(1,2,3)) + targets.sum(dim=(1,2,3))
    dice = (2. * intersection + 1e-8) / (union + 1e-8)
    return dice.mean().item()

# IoU
def iou_score(preds, targets, threshold=0.5):
    preds = torch.sigmoid(preds)
    preds = (preds > threshold).float()
    intersection = (preds * targets).sum(dim=(1,2,3))
    union = preds.sum(dim=(1,2,3)) + targets.sum(dim=(1,2,3)) - intersection
    iou = (intersection + 1e-8) / (union + 1e-8)
    return iou.mean().item()

# Accuracy
def pixel_accuracy(preds, targets, threshold=0.5):
    preds = torch.sigmoid(preds)
    preds = (preds > threshold).float()
    correct = (preds == targets).float()
    return correct.mean().item()

# Entraînement
def Train(image_path, mask_path, image_path_test, mask_path_test, num_epochs=20):
    X_train = np.load(image_path)
    Y_train = np.load(mask_path)
    X_test = np.load(image_path_test)
    Y_test = np.load(mask_path_test)

    print(f"Training set: {X_train.shape}, {Y_train.shape}")
    print(f"Test set: {X_test.shape}, {Y_test.shape}")
    print(f"Positive ratio in training: {Y_train.mean():.4f}")

    train_dataset = LungDataset(X_train, Y_train)
    test_dataset = LungDataset(X_test, Y_test)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=2)

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

    model = SwinUnet(config=config, img_size=256, num_classes=1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    pos_weight = (Y_train == 0).sum() / (Y_train == 1).sum()
    print(f"Positive weight: {pos_weight:.2f}")
    
    criterion = DiceBCELoss(pos_weight=torch.tensor([pos_weight]).to(device))
    optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    best_dice = 0.0
    patience_counter = 0
    patience_limit = 10

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)

        model.eval()
        dice_total = 0.0
        iou_total = 0.0
        acc_total = 0.0
        test_loss = 0.0
        
        with torch.no_grad():
            for images, masks in test_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                test_loss += criterion(outputs, masks).item()
                dice_total += dice_score(outputs, masks)
                iou_total += iou_score(outputs, masks)
                acc_total += pixel_accuracy(outputs, masks)

        avg_test_loss = test_loss / len(test_loader)
        dice_avg = dice_total / len(test_loader)
        iou_avg = iou_total / len(test_loader)
        acc_avg = acc_total / len(test_loader)

        scheduler.step(avg_test_loss)
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch [{epoch+1}/{num_epochs}] - "
              f"Loss: {avg_loss:.4f} | Test Loss: {avg_test_loss:.4f} | "
              f"Dice: {dice_avg:.4f} | IoU: {iou_avg:.4f} | Acc: {acc_avg:.4f} | "
              f"LR: {current_lr:.2e}")

        if dice_avg > best_dice:
            best_dice = dice_avg
            torch.save(model.state_dict(), "best_swin_unet_lung.pth")
            patience_counter = 0
            print(f"Meilleur modèle sauvegardé (Dice: {best_dice:.4f})")
        else:
            patience_counter += 1

        if patience_counter >= patience_limit:
            print(f"Arrêt anticipé après {epoch+1} epochs")
            break

    torch.save(model.state_dict(), "final_swin_unet_lung.pth")
    print(f"Entraînement terminé. Meilleur Dice: {best_dice:.4f}")
    print("Modèle final sauvegardé sous 'final_swin_unet_lung.pth'")

# Test final
def test_model(model_path, image_path_test, mask_path_test):
    X_test = np.load(image_path_test)
    Y_test = np.load(mask_path_test)

    test_dataset = LungDataset(X_test, Y_test)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

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

    model = SwinUnet(config=config, img_size=256, num_classes=1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.load_state_dict(torch.load(model_path))

    model.eval()
    dice_total = 0.0
    iou_total = 0.0
    acc_total = 0.0

    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            dice_total += dice_score(outputs, masks)
            iou_total += iou_score(outputs, masks)
            acc_total += pixel_accuracy(outputs, masks)

    dice_avg = dice_total / len(test_loader)
    iou_avg = iou_total / len(test_loader)
    acc_avg = acc_total / len(test_loader)

    print(f"Test final - Dice: {dice_avg:.4f} | IoU: {iou_avg:.4f} | Acc: {acc_avg:.4f}")
    return dice_avg, iou_avg, acc_avg
