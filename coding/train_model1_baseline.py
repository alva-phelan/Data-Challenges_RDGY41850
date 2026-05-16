#imports (taken from example notebook)
import os, glob, random, math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import random_split

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

#things to try:
#change loss
#try windowing (ITK slice to visualise)

# nvidia-smi -> need to fully use the memory on the gpu
# ssh into the sonic node that is running, then nvidia

#config
DATA_DIR = "./ct_rate_data/dataset/train_fixed"
LABEL_FILE ="./ct_rate_data/dataset/multi_abnormality_labels/train_predicted_labels.csv"
#can edit these later
BATCH_SIZE = 8 #increase to use gpu
EPOCHS = 30
TRAIN_SIZE = 1000
LR = 1e-4
NUM_WORKERS = 1 #pin_memory=true
SEED = 12345

#device and versions
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

set_determinism(SEED)

#dataset class
class CTRateDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        print("Loading labels from CSV...")
        full_df = pd.read_csv(csv_file)
        
        self.root_dir = root_dir
        self.transform = transform

        self.label_cols = [
            "Medical material", "Arterial wall calcification", "Cardiomegaly", "Pericardial effusion",
            "Coronary artery wall calcification", "Hiatal hernia", "Lymphadenopathy", "Emphysema",
            "Atelectasis", "Lung nodule", "Lung opacity", "Pulmonary fibrotic sequela", "Pleural effusion",
            "Mosaic attenuation pattern", "Peribronchial thickening", "Consolidation", "Bronchiectasis",
            "Interlobular septal thickening"
        ]

        print("Scanning hard drive for valid images. This may take a few seconds...")
        valid_records = []
        
        # check each row in the CSV to see if the file physically exists
        for idx, row in full_df.iterrows():
            raw_volume_name = row['VolumeName']
            base_name = raw_volume_name.replace('.nii.gz', '')  
            
            parts = base_name.split('_')
            folder_prefix = f"{parts[0]}_{parts[1]}"
            patient_folder = base_name.rsplit('_', 1)[0] 
            
            path_fixed = f"./ct_rate_data/dataset/train_fixed/{folder_prefix}/{patient_folder}/{raw_volume_name}"
            path_normal = f"./ct_rate_data/dataset/train/{folder_prefix}/{patient_folder}/{raw_volume_name}"
            
            # If we find the file, record its exact true path
            if os.path.exists(path_fixed):
                row_dict = row.to_dict()
                row_dict['true_path'] = path_fixed
                valid_records.append(row_dict)
            elif os.path.exists(path_normal):
                row_dict = row.to_dict()
                row_dict['true_path'] = path_normal
                valid_records.append(row_dict)
                
            # Once we find 100 valid patients that ACTUALLY exist, stop searching!
            if len(valid_records) >= TRAIN_SIZE:
                break
                
        self.labels_df = pd.DataFrame(valid_records)
        print(f"Success! Found {len(self.labels_df)} valid CT scans for training.")

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        row = self.labels_df.iloc[idx]
        img_path = row['true_path'] 
        
        labels = row[self.label_cols].values.astype('float32')
        sample = {"image": img_path, "labels": labels}

        if self.transform:
            sample = self.transform(sample)

        return sample
    
#monai transforms
transforms = Compose([
    LoadImaged(keys=["image"]),
    EnsureChannelFirstd(keys=["image"]),
    Orientationd(keys=["image"], axcodes="RAS"),
    Spacingd(keys=["image"], pixdim=(2.0, 2.0, 2.0), mode="bilinear"),
    #augmentations
    RandFlipd(keys=["image"], prob = 0.5, spatial_axis=0),
    RandAffined(keys=["image"], prob = 0.5, rotate_range=(0.1, 0.1, 0.1), scale_range=(0.1,0.1,0.1)),
    Resized(keys=["image"], spatial_size =(128, 128, 128)),
    ScaleIntensityRanged(keys=["image"], a_min=-1000, a_max=400, b_min=0.0, b_max=1.0, clip=True),
    EnsureTyped(keys=["image", "labels"])
])      

#initialize data 
dataset = CTRateDataset(csv_file=LABEL_FILE, root_dir=DATA_DIR, transform=transforms)
#test and val split
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_records, val_records = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_records, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory = True)
val_loader = DataLoader(val_records, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory = True)

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
    train_indices = train_records.indices
    #pull only label columns for those specific training rows 
    train_labels_df = dataset.labels_df.iloc[train_indices][dataset.label_cols]
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
            #if batch_idx % 20 == 0:
                #print(f"Epoch {epoch+1}: Batch {batch_idx}/{len(train_loader)} processed...")

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
       
        