#imports 
import os, glob, random, math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import random_split
from sklearn.model_selection import GroupShuffleSplit

import monai
from monai.utils import set_determinism
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
    CropForegroundd, ResizeWithPadOrCropd, RandFlipd, RandAffined, 
    RandGaussianNoised, RandScaleIntensityd, RandShiftIntensityd,
    EnsureTyped, ScaleIntensityRanged, Resized
)
from monai.data import Dataset, DataLoader, PersistentDataset
from monai.networks.nets import resnet50
from monai.inferers import SimpleInferer

from torchmetrics.classification import MultilabelAUROC, MultilabelF1Score, MultilabelAveragePrecision

#config
DATA_DIR = "./ct_rate_data/dataset/train_fixed"
LABEL_FILE ="./ct_rate_data/dataset/multi_abnormality_labels/train_predicted_labels.csv"
#can edit these later
BATCH_SIZE = 4 
EPOCHS = 30
LR = 1e-4
NUM_WORKERS = 4 
SEED = 12345
total_scans = 1000
max_volumes_per_patient = 2
train_val_test_split = [0.7, 0.15, 0.15]

#device and versions
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

set_determinism(SEED)

# Find valid files and cap per patient 
raw_df = pd.read_csv(LABEL_FILE)
valid_records = []
patient_counts = {}

print(f"Finding {total_scans} valid volumes...")
for _, row in raw_df.iterrows():
    vol_name = row['VolumeName']
    pid = vol_name.rsplit('_', 1)[0]
    
    # Cap volumes per patient
    if patient_counts.get(pid, 0) >= max_volumes_per_patient:
        continue
        
    # find the file path
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

# 3-Way Patient-level Split
# Split test out first
gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=SEED) # 30% for Val+Test
train_idx, val_test_idx = next(gss.split(master_df, groups=master_df['PatientID']))
train_df = master_df.iloc[train_idx]
val_test_df = master_df.iloc[val_test_idx]

# Split Val and Test
gss_sub = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=SEED)
val_idx, test_idx = next(gss_sub.split(val_test_df, groups=val_test_df['PatientID']))
val_df = val_test_df.iloc[val_idx]
test_df = val_test_df.iloc[test_idx]

# save test set for separate script
test_df.to_csv("test_patients_locked.csv", index=False)
print(f"Final Counts -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

# Dataset & Dataloaders 
class CTRateDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        self.label_cols = ["Medical material", "Arterial wall calcification", "Cardiomegaly", "Pericardial effusion", "Coronary artery wall calcification", "Hiatal hernia", "Lymphadenopathy", "Emphysema", "Atelectasis", "Lung nodule", "Lung opacity", "Pulmonary fibrotic sequela", "Pleural effusion", "Mosaic attenuation pattern", "Peribronchial thickening", "Consolidation", "Bronchiectasis", "Interlobular septal thickening"]
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sample = {"image": row['image_path'], "labels": row[self.label_cols].values.astype('float32')}
        return self.transform(sample) if self.transform else sample
    
# Define Separate Transforms 
# Training: Includes Augmentations
train_transforms = Compose([
    LoadImaged(keys=["image"]),
    EnsureChannelFirstd(keys=["image"]),
    Orientationd(keys=["image"], axcodes="RAS"),
    Spacingd(keys=["image"], pixdim=(2.0, 2.0, 2.0), mode="bilinear"),
    Resized(keys=["image"], spatial_size=(128, 128, 128)),
    ScaleIntensityRanged(keys=["image"], a_min=-1000, a_max=400, b_min=0.0, b_max=1.0, clip=True),
    RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
    RandAffined(keys=["image"], prob=0.5, rotate_range=(0.1, 0.1, 0.1), translate_range=(10, 10, 10)),
    EnsureTyped(keys=["image", "labels"])
])

# Validation/Testing: Clean (No Flips or Rotations)
val_transforms = Compose([
    LoadImaged(keys=["image"]),
    EnsureChannelFirstd(keys=["image"]),
    Orientationd(keys=["image"], axcodes="RAS"),
    Spacingd(keys=["image"], pixdim=(2.0, 2.0, 2.0), mode="bilinear"),
    Resized(keys=["image"], spatial_size=(128, 128, 128)),
    ScaleIntensityRanged(keys=["image"], a_min=-1000, a_max=400, b_min=0.0, b_max=1.0, clip=True),
    EnsureTyped(keys=["image", "labels"])
])

train_ds = CTRateDataset(train_df, transform=train_transforms)
val_ds = CTRateDataset(val_df, transform=val_transforms)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)


print(f"Building 3D ResNet50 on: {device}")

model = resnet50(
    spatial_dims=3,
    n_input_channels=1,
    num_classes=18
    ).to(device)

# metrics for validation ("macro" ensures rare diseases matter equally)
auroc_metric = MultilabelAUROC(num_labels=18, average="macro").to(device)
f1_metric = MultilabelF1Score(num_labels=18, average="macro").to(device)
pr_auc_metric = MultilabelAveragePrecision(num_labels=18, average="macro").to(device)

#training
def train():

    #row numbers used for training split
    train_labels_df = train_ds.df[train_ds.label_cols]
    #convert to python tensor
    all_labels = torch.tensor(train_labels_df.values, dtype=torch.float32)

    positive_counts = all_labels.sum(dim=0)
    negative_counts = len(all_labels) - positive_counts
    
    # weight = (no of negatives) / (no of positives)
    # adding 1e-6 prevents division by zero if a class has 0 positives in a small batch
    pos_weights = negative_counts / (positive_counts + 1e-6)
    pos_weights = pos_weights.to(device)

    print(f"Calculated Class Weights: {pos_weights}")

    #try categorical, sparse, weighted cross entropy
    criterion = nn.BCEWithLogitsLoss(pos_weight = pos_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    best_val_loss = float('inf')


    print("Starting Training...")

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for batch_idx, batch_data in enumerate(train_loader):
            inputs = batch_data["image"].to(device)
            labels = batch_data["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

            #heartbeat to see if running
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}: Batch {batch_idx}/{len(train_loader)} processed...")

        avg_train_loss = running_loss/len(train_loader)

        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f}")

        #validation
        model.eval()
        val_loss = 0.0

        # reset eval metrics at the start of each validation epoch
        auroc_metric.reset()
        f1_metric.reset()
        pr_auc_metric.reset()

        with torch.no_grad():
            for batch_data in val_loader:
                inputs = batch_data["image"].to(device)
                labels = batch_data["labels"].to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()

                # update metrics (convert outputs to probabilities using sigmoid)
                probs = torch.sigmoid(outputs)
                probs_clean = probs.as_tensor()
                labels_clean = labels.as_tensor().long()
                auroc_metric.update(probs_clean, labels_clean)
                f1_metric.update(probs_clean, labels_clean)
                pr_auc_metric.update(probs_clean, labels_clean)

        avg_val_loss = val_loss / len(val_loader)

        # compute final metric scores
        epoch_auroc = auroc_metric.compute().item()
        epoch_f1 = f1_metric.compute().item()
        epoch_prauc = pr_auc_metric.compute().item()

        print(f"Epoch {epoch+1}/{EPOCHS} | Val Loss: {avg_val_loss:.4f} | AUROC: {epoch_auroc:.4f} | PR-AUC: {epoch_prauc:.4f} | F1: {epoch_f1:.4f}")

        if avg_val_loss < best_val_loss:
            print(f"Val loss improved from {best_val_loss:.4f} to {avg_val_loss:.4f}. Saving model")
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "resnet50_baseline_best.pth")

    print("Training Complete. Model saved to resnet50_baseline_best.pth")

if __name__ == "__main__":
    train()
       
        

