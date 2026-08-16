import os
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import GroupShuffleSplit

from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
    Resized, EnsureTyped, ScaleIntensityRanged
)
from monai.data import Dataset, DataLoader

import torchvision
from torchvision.models import ResNet50_Weights

from torchmetrics.classification import MultilabelAUROC, MultilabelF1Score, MultilabelAveragePrecision

from preprocessing_multiwindow import MultiWindowIntensityd, volume_to_multiwindow_slices

# Config - match the training run being evaluated 
DATA_DIR = os.environ.get("CT_DATA_DIR", "./ct_rate_data/dataset/train_fixed")
LABEL_FILE = os.environ.get("CT_LABEL_FILE", "./ct_rate_data/dataset/multi_abnormality_labels/train_predicted_labels.csv")
SEED = int(os.environ.get("SEED", 12345))
total_scans = int(os.environ.get("TOTAL_SCANS", 1000))
max_volumes_per_patient = int(os.environ.get("MAX_VOLUMES_PER_PATIENT", 2))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", 4))

NUM_SLICES = int(os.environ.get("NUM_SLICES", 32))
FREEZE_BACKBONE = os.environ.get("FREEZE_BACKBONE", "True") == "True"
POOLING = os.environ.get("POOLING", "attention")
USE_MULTIWINDOW = os.environ.get("USE_MULTIWINDOW", "True") == "True"
SLICE_SIZE = 224
VOLUME_SIZE = (128, 128, 128)
SINGLE_WINDOW_HU = (-1000, 400)

window_tag = "multiwindow" if USE_MULTIWINDOW else "singlewindow"
CHECKPOINT_PATH = os.environ.get(
    "CHECKPOINT_PATH",
    f"slice2d_{POOLING}_frozen{FREEZE_BACKBONE}_k{NUM_SLICES}_{window_tag}_best.pth"
)

print(f"Config: NUM_SLICES={NUM_SLICES}, POOLING={POOLING}, USE_MULTIWINDOW={USE_MULTIWINDOW}, "
      f"checkpoint={CHECKPOINT_PATH}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABEL_COLS = ["Medical material", "Arterial wall calcification", "Cardiomegaly", "Pericardial effusion",
              "Coronary artery wall calcification", "Hiatal hernia", "Lymphadenopathy", "Emphysema",
              "Atelectasis", "Lung nodule", "Lung opacity", "Pulmonary fibrotic sequela", "Pleural effusion",
              "Mosaic attenuation pattern", "Peribronchial thickening", "Consolidation", "Bronchiectasis",
              "Interlobular septal thickening"]

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def volume_to_slices(volume: torch.Tensor) -> torch.Tensor:
    if USE_MULTIWINDOW:
        return volume_to_multiwindow_slices(volume, NUM_SLICES, SLICE_SIZE, IMAGENET_MEAN, IMAGENET_STD)
    depth = volume.shape[1]
    idxs = torch.linspace(0, depth - 1, NUM_SLICES).long()
    slices = volume[0, idxs, :, :]
    slices = slices.unsqueeze(1).repeat(1, 3, 1, 1)
    slices = torch.nn.functional.interpolate(
        slices, size=(SLICE_SIZE, SLICE_SIZE), mode="bilinear", align_corners=False
    )
    slices = (slices - IMAGENET_MEAN) / IMAGENET_STD
    return slices


# Reconstruct the exact same leakage-free split used in training
print(f"Finding {total_scans} valid volumes (reconstructing training-time split)...")
raw_df = pd.read_csv(LABEL_FILE)
valid_records = []
patient_counts = {}

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
val_test_df = master_df.iloc[val_test_idx]

gss_sub = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=SEED)
val_idx, test_idx = next(gss_sub.split(val_test_df, groups=val_test_df['PatientID']))
test_df = val_test_df.iloc[test_idx]

print(f"Reconstructed test set: {len(test_df)} scans")


class CTRateSliceTestDataset(Dataset):
    def __init__(self, df, transform):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sample = {"image": row['image_path']}
        sample = self.transform(sample)
        labels = torch.tensor(row[LABEL_COLS].values.astype('float32'))
        volume = sample["image"]
        if hasattr(volume, "as_tensor"):
            volume = volume.as_tensor()
        slices = volume_to_slices(volume)
        return {"image": slices, "labels": labels}


if USE_MULTIWINDOW:
    _windowing_transform = MultiWindowIntensityd(keys=["image"])
else:
    a_min, a_max = SINGLE_WINDOW_HU
    _windowing_transform = ScaleIntensityRanged(keys=["image"], a_min=a_min, a_max=a_max, b_min=0.0, b_max=1.0, clip=True)

eval_transforms = Compose([
    LoadImaged(keys=["image"]),
    EnsureChannelFirstd(keys=["image"]),
    Orientationd(keys=["image"], axcodes="RAS"),
    Spacingd(keys=["image"], pixdim=(2.0, 2.0, 2.0), mode="bilinear"),
    Resized(keys=["image"], spatial_size=VOLUME_SIZE),
    _windowing_transform,
    EnsureTyped(keys=["image"]),
])

test_ds = CTRateSliceTestDataset(test_df, eval_transforms)
test_loader = DataLoader(test_ds, batch_size=1, num_workers=NUM_WORKERS, shuffle=False)


# Model (identical definitions to train_model4_pretrained2d_fm.py)
class SliceEncoder(nn.Module):
    def __init__(self, freeze_backbone=True):
        super().__init__()
        backbone = torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.out_dim = 2048
        if freeze_backbone:
            for p in self.features.parameters():
                p.requires_grad = False

    def forward(self, x):
        return self.features(x).flatten(1)


class AttentionPool(nn.Module):
    def __init__(self, dim=2048, hidden=256):
        super().__init__()
        self.attn = nn.Sequential(nn.Linear(dim, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, feats):
        scores = self.attn(feats)
        weights = torch.softmax(scores, dim=1)
        pooled = (weights * feats).sum(dim=1)
        return pooled, weights.squeeze(-1)


class Slice2DClassifier(nn.Module):
    def __init__(self, num_classes=18, freeze_backbone=True, pooling="attention"):
        super().__init__()
        self.encoder = SliceEncoder(freeze_backbone)
        self.pooling_type = pooling
        if pooling == "attention":
            self.pool = AttentionPool(self.encoder.out_dim, 256)
        self.classifier = nn.Linear(self.encoder.out_dim, num_classes)

    def forward(self, x):
        B, K, C, H, W = x.shape
        x = x.view(B * K, C, H, W)
        feats = self.encoder(x).view(B, K, -1)
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


def evaluate():
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Error: {CHECKPOINT_PATH} not found. Has training finished / saved a checkpoint yet?")
        return

    print(f"Loading saved model from {CHECKPOINT_PATH}...")
    model = Slice2DClassifier(num_classes=18, freeze_backbone=FREEZE_BACKBONE, pooling=POOLING).to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()

    auroc_metric = MultilabelAUROC(num_labels=18, average="macro").to(device)
    f1_metric = MultilabelF1Score(num_labels=18, average="macro").to(device)
    pr_auc_metric = MultilabelAveragePrecision(num_labels=18, average="macro").to(device)

    # per-class versions (average=None), for the Section 5 "per-class
    # performance" table the writing instructions ask for -- costs nothing
    # extra since it's computed from the same predictions
    auroc_per_class = MultilabelAUROC(num_labels=18, average=None).to(device)
    f1_per_class = MultilabelF1Score(num_labels=18, average=None).to(device)

    print("Starting evaluation scan...")
    with torch.no_grad():
        for batch_data in test_loader:
            inputs = batch_data["image"].to(device)
            labels = batch_data["labels"].to(device)

            outputs, _ = model(inputs)
            probabilities = torch.sigmoid(outputs)

            clean_probs = probabilities.as_tensor() if hasattr(probabilities, "as_tensor") else probabilities
            clean_labels = labels.as_tensor().long() if hasattr(labels, "as_tensor") else labels.long()

            auroc_metric.update(clean_probs, clean_labels)
            f1_metric.update(clean_probs, clean_labels)
            pr_auc_metric.update(clean_probs, clean_labels)
            auroc_per_class.update(clean_probs, clean_labels)
            f1_per_class.update(clean_probs, clean_labels)

    final_auroc = auroc_metric.compute().item()
    final_f1 = f1_metric.compute().item()
    final_prauc = pr_auc_metric.compute().item()
    auroc_by_class = auroc_per_class.compute().cpu().tolist()
    f1_by_class = f1_per_class.compute().cpu().tolist()

    print("\n" + "=" * 35)
    print(f"Model 4 ({window_tag}) | Final test results | "
          f"AUROC: {final_auroc:.4f} | PR-AUC: {final_prauc:.4f} | F1-Score: {final_f1:.4f}")

    print("\nPer-class breakdown:")
    print(f"{'Label':<40} {'AUROC':>8} {'F1':>8}")
    for label, a, f in zip(LABEL_COLS, auroc_by_class, f1_by_class):
        print(f"{label:<40} {a:>8.4f} {f:>8.4f}")


if __name__ == "__main__":
    evaluate()