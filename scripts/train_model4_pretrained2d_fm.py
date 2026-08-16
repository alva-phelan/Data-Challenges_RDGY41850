import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import GroupShuffleSplit
import monai
from monai.utils import set_determinism
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
    Resized, RandFlipd, RandAffined, EnsureTyped, ScaleIntensityRanged
)
from monai.data import Dataset, DataLoader

import torchvision
from torchvision.models import ResNet50_Weights

from torchmetrics.classification import MultilabelAUROC, MultilabelF1Score, MultilabelAveragePrecision

from preprocessing_multiwindow import MultiWindowIntensityd, volume_to_multiwindow_slices

# Config (all overridable via environment variables, so a SLURM script can
# run different ablation arms without editing this file each time)
DATA_DIR = os.environ.get("CT_DATA_DIR", "./ct_rate_data/dataset/train_fixed")
LABEL_FILE = os.environ.get("CT_LABEL_FILE", "./ct_rate_data/dataset/multi_abnormality_labels/train_predicted_labels.csv")

BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 4))
EPOCHS = int(os.environ.get("EPOCHS", 30))
LR = float(os.environ.get("LR", 1e-4))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", 4))
SEED = int(os.environ.get("SEED", 12345))
total_scans = int(os.environ.get("TOTAL_SCANS", 1000))
max_volumes_per_patient = int(os.environ.get("MAX_VOLUMES_PER_PATIENT", 2))

# Model 4 specific config / ablation switches
NUM_SLICES = int(os.environ.get("NUM_SLICES", 32))          # try 16 / 32 / 64 for the ablation table
FREEZE_BACKBONE = os.environ.get("FREEZE_BACKBONE", "True") == "True"  # try True / False
POOLING = os.environ.get("POOLING", "attention")             # "attention" / "mean" / "max"
SLICE_SIZE = 224         # ResNet50 native input resolution
VOLUME_SIZE = (128, 128, 128)  # same as Models 1-3, sliced along last axis
USE_MULTIWINDOW = os.environ.get("USE_MULTIWINDOW", "True") == "True"
                          # True = 3 clinical windows as channels (new),
                          # False = single window replicated x3 (old behaviour,
                          # kept for the ablation comparing the two)
SINGLE_WINDOW_HU = (-1000, 400)  # only used when USE_MULTIWINDOW is False

print(f"Config: NUM_SLICES={NUM_SLICES}, FREEZE_BACKBONE={FREEZE_BACKBONE}, "
      f"POOLING={POOLING}, USE_MULTIWINDOW={USE_MULTIWINDOW}, EPOCHS={EPOCHS}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_determinism(SEED)

# Data: identical leakage-free split logic to Model 2
raw_df = pd.read_csv(LABEL_FILE)
valid_records = []
patient_counts = {}

print(f"Finding {total_scans} valid volumes...")
for _, row in raw_df.iterrows():
    vol_name = row['VolumeName']
    pid = vol_name.rsplit('_', 1)[0]

    if patient_counts.get(pid, 0) >= max_volumes_per_patient:
        continue

    base = vol_name.replace('.nii.gz', '')
    prefix = f"{base.split('_')[0]}_{base.split('_')[1]}"
    p_folder = base.rsplit('_', 1)[0]

    path_fixed = os.path.join(DATA_DIR, prefix, p_folder, vol_name)
    path_orig = os.path.join("./ct_rate_data/dataset/train", prefix, p_folder, vol_name)
    actual_path = path_fixed if os.path.exists(path_fixed) else (path_orig if os.path.exists(path_orig) else None)

    if actual_path:
        row_dict = row.to_dict()
        row_dict['image_path'] = actual_path
        row_dict['PatientID'] = pid
        valid_records.append(row_dict)
        patient_counts[pid] = patient_counts.get(pid, 0) + 1

    if len(valid_records) >= total_scans:
        break

master_df = pd.DataFrame(valid_records)

gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=SEED)
train_idx, val_test_idx = next(gss.split(master_df, groups=master_df['PatientID']))
train_df = master_df.iloc[train_idx]
val_test_df = master_df.iloc[val_test_idx]

gss_sub = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=SEED)
val_idx, test_idx = next(gss_sub.split(val_test_df, groups=val_test_df['PatientID']))
val_df = val_test_df.iloc[val_idx]
test_df = val_test_df.iloc[test_idx]

print(f"Final Counts -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

LABEL_COLS = ["Medical material", "Arterial wall calcification", "Cardiomegaly", "Pericardial effusion",
              "Coronary artery wall calcification", "Hiatal hernia", "Lymphadenopathy", "Emphysema",
              "Atelectasis", "Lung nodule", "Lung opacity", "Pulmonary fibrotic sequela", "Pleural effusion",
              "Mosaic attenuation pattern", "Peribronchial thickening", "Consolidation", "Bronchiectasis",
              "Interlobular septal thickening"]

# ImageNet normalisation stats, applied after our own HU windowing to [0,1]
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def volume_to_slices(volume: torch.Tensor) -> torch.Tensor:

    if USE_MULTIWINDOW:
        return volume_to_multiwindow_slices(volume, NUM_SLICES, SLICE_SIZE, IMAGENET_MEAN, IMAGENET_STD)

    depth = volume.shape[1]
    idxs = torch.linspace(0, depth - 1, NUM_SLICES).long()
    slices = volume[0, idxs, :, :]  # [K, H, W]
    slices = slices.unsqueeze(1).repeat(1, 3, 1, 1)  # [K, 3, H, W]
    slices = torch.nn.functional.interpolate(
        slices, size=(SLICE_SIZE, SLICE_SIZE), mode="bilinear", align_corners=False
    )
    slices = (slices - IMAGENET_MEAN) / IMAGENET_STD
    return slices


class CTRateSliceDataset(Dataset):
    def __init__(self, df, transform, augment=False):
        self.df = df
        self.transform = transform
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sample = {"image": row['image_path']}
        sample = self.transform(sample)
        labels = torch.tensor(row[LABEL_COLS].values.astype('float32'))
        volume = sample["image"]
        if hasattr(volume, "as_tensor"):  # strip MONAI MetaTensor metadata before slicing
            volume = volume.as_tensor()
        slices = volume_to_slices(volume)
        return {"image": slices, "labels": labels}


# Windowing step is the only thing that differs between the two ablation arms.
if USE_MULTIWINDOW:
    _windowing_transform = MultiWindowIntensityd(keys=["image"])
else:
    a_min, a_max = SINGLE_WINDOW_HU
    _windowing_transform = ScaleIntensityRanged(keys=["image"], a_min=a_min, a_max=a_max, b_min=0.0, b_max=1.0, clip=True)

# Volume-level preprocessing only (slicing + slice-level augmentation happens above)
train_volume_transforms = Compose([
    LoadImaged(keys=["image"]),
    EnsureChannelFirstd(keys=["image"]),
    Orientationd(keys=["image"], axcodes="RAS"),
    Spacingd(keys=["image"], pixdim=(2.0, 2.0, 2.0), mode="bilinear"),
    Resized(keys=["image"], spatial_size=VOLUME_SIZE),
    _windowing_transform,
    RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
    RandAffined(keys=["image"], prob=0.5, rotate_range=(0.1, 0.1, 0.1), translate_range=(10, 10, 10)),
    EnsureTyped(keys=["image"]),
])

val_volume_transforms = Compose([
    LoadImaged(keys=["image"]),
    EnsureChannelFirstd(keys=["image"]),
    Orientationd(keys=["image"], axcodes="RAS"),
    Spacingd(keys=["image"], pixdim=(2.0, 2.0, 2.0), mode="bilinear"),
    Resized(keys=["image"], spatial_size=VOLUME_SIZE),
    _windowing_transform,
    EnsureTyped(keys=["image"]),
])

train_ds = CTRateSliceDataset(train_df, train_volume_transforms, augment=True)
val_ds = CTRateSliceDataset(val_df, val_volume_transforms, augment=False)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)


# Model: shared 2D backbone + attention pooling over slices + linear head
class SliceEncoder(nn.Module):
    def __init__(self, freeze_backbone=True):
        super().__init__()
        backbone = torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.features = nn.Sequential(*list(backbone.children())[:-1])  # drop fc -> [B,2048,1,1]
        self.out_dim = 2048
        if freeze_backbone:
            for p in self.features.parameters():
                p.requires_grad = False

    def forward(self, x):  # x: [B, 3, H, W]
        feat = self.features(x)
        return feat.flatten(1)  # [B, 2048]


class AttentionPool(nn.Module):
    def __init__(self, dim=2048, hidden=256):
        super().__init__()
        self.attn = nn.Sequential(nn.Linear(dim, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, feats):  # feats: [B, K, dim]
        scores = self.attn(feats)                  # [B, K, 1]
        weights = torch.softmax(scores, dim=1)      # [B, K, 1]
        pooled = (weights * feats).sum(dim=1)       # [B, dim]
        return pooled, weights.squeeze(-1)          # weights useful for interpretability figures


class Slice2DClassifier(nn.Module):
    def __init__(self, num_classes=18, freeze_backbone=True, pooling="attention"):
        super().__init__()
        self.encoder = SliceEncoder(freeze_backbone)
        self.pooling_type = pooling
        if pooling == "attention":
            self.pool = AttentionPool(self.encoder.out_dim, 256)
        self.classifier = nn.Linear(self.encoder.out_dim, num_classes)

    def forward(self, x):  # x: [B, K, 3, H, W]
        B, K, C, H, W = x.shape
        x = x.view(B * K, C, H, W)
        feats = self.encoder(x).view(B, K, -1)  # [B, K, dim]

        if self.pooling_type == "attention":
            pooled, attn_weights = self.pool(feats)
        elif self.pooling_type == "mean":
            pooled, attn_weights = feats.mean(dim=1), None
        elif self.pooling_type == "max":
            pooled, attn_weights = feats.max(dim=1).values, None
        else:
            raise ValueError(f"Unknown pooling type: {self.pooling_type}")

        logits = self.classifier(pooled)
        return logits, attn_weights


class PerLabelWindowGatedClassifier(nn.Module):
    def __init__(self, num_classes=18, num_windows=3, freeze_backbone=True):
        super().__init__()
        self.encoder = SliceEncoder(freeze_backbone)  # shared across windows
        self.slice_pool = AttentionPool(self.encoder.out_dim, 256)  # pools over K slices, per window
        self.num_windows = num_windows
        self.num_classes = num_classes
        # one gate score per (label, window) pair, computed from the pooled window features
        self.gate = nn.Linear(self.encoder.out_dim, num_classes)
        self.classifier = nn.Linear(self.encoder.out_dim, num_classes)

    def forward(self, x):  # x: [B, num_windows, K, 3, H, W] -- 3 windows, each K slices, each replicated to 3ch
        B, Wn, K, C, H, W = x.shape
        window_feats = []  # each [B, dim]
        for w in range(Wn):
            slices_w = x[:, w].reshape(B * K, C, H, W)
            feats = self.encoder(slices_w).view(B, K, -1)
            pooled_w, _ = self.slice_pool(feats)  # [B, dim]
            window_feats.append(pooled_w)
        window_feats = torch.stack(window_feats, dim=1)  # [B, num_windows, dim]

        # per-label gate: how much does label c rely on window w, for this sample
        gate_logits = self.gate(window_feats)              # [B, num_windows, num_classes]
        gate_weights = torch.softmax(gate_logits, dim=1)   # softmax over windows, per label

        # weighted combination of window features, separately per label:
        # [B, num_windows, num_classes, 1] * [B, num_windows, 1, dim] -> sum over windows -> [B, num_classes, dim]
        combined = (gate_weights.unsqueeze(-1) * window_feats.unsqueeze(2)).sum(dim=1)

        logits = (combined * self.classifier.weight.unsqueeze(0)).sum(-1) + self.classifier.bias
        return logits, gate_weights.mean(dim=0)  # batch-averaged [num_windows, num_classes] for the heatmap


print(f"Building Slice2DClassifier (freeze_backbone={FREEZE_BACKBONE}, pooling={POOLING}) on: {device}")
model = Slice2DClassifier(num_classes=18, freeze_backbone=FREEZE_BACKBONE, pooling=POOLING).to(device)

auroc_metric = MultilabelAUROC(num_labels=18, average="macro").to(device)
f1_metric = MultilabelF1Score(num_labels=18, average="macro").to(device)
pr_auc_metric = MultilabelAveragePrecision(num_labels=18, average="macro").to(device)


def train():
    train_labels_df = train_ds.df[LABEL_COLS]
    all_labels = torch.tensor(train_labels_df.values, dtype=torch.float32)
    positive_counts = all_labels.sum(dim=0)
    negative_counts = len(all_labels) - positive_counts
    pos_weights = (negative_counts / (positive_counts + 1e-6)).to(device)

    print(f"Calculated Class Weights: {pos_weights}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    # Only optimise parameters that require grad (matters when backbone is frozen)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)

    best_val_loss = float('inf')

    print("Starting Training...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for batch_idx, batch_data in enumerate(train_loader):
            inputs = batch_data["image"].to(device)
            labels = batch_data["labels"].to(device)

            optimizer.zero_grad()
            outputs, _ = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}: Batch {batch_idx}/{len(train_loader)} processed...")

        avg_train_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f}")

        model.eval()
        val_loss = 0.0
        auroc_metric.reset(); f1_metric.reset(); pr_auc_metric.reset()

        with torch.no_grad():
            for batch_data in val_loader:
                inputs = batch_data["image"].to(device)
                labels = batch_data["labels"].to(device)

                outputs, _ = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                probs = torch.sigmoid(outputs)
                probs_clean = probs.as_tensor() if hasattr(probs, "as_tensor") else probs
                labels_clean = labels.as_tensor().long() if hasattr(labels, "as_tensor") else labels.long()
                auroc_metric.update(probs_clean, labels_clean)
                f1_metric.update(probs_clean, labels_clean)
                pr_auc_metric.update(probs_clean, labels_clean)

        avg_val_loss = val_loss / len(val_loader)
        epoch_auroc = auroc_metric.compute().item()
        epoch_f1 = f1_metric.compute().item()
        epoch_prauc = pr_auc_metric.compute().item()

        print(f"Epoch {epoch+1}/{EPOCHS} | Val Loss: {avg_val_loss:.4f} | "
              f"AUROC: {epoch_auroc:.4f} | PR-AUC: {epoch_prauc:.4f} | F1: {epoch_f1:.4f}")

        if avg_val_loss < best_val_loss:
            print(f"Val loss improved from {best_val_loss:.4f} to {avg_val_loss:.4f}. Saving model")
            best_val_loss = avg_val_loss
            window_tag = "multiwindow" if USE_MULTIWINDOW else "singlewindow"
            torch.save(model.state_dict(),
                       f"slice2d_{POOLING}_frozen{FREEZE_BACKBONE}_k{NUM_SLICES}_{window_tag}_best.pth")

    print("Training Complete.")


if __name__ == "__main__":
    train()