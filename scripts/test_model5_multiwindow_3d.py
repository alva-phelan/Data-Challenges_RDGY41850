import os
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit

from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
    Resized, EnsureTyped
)
from monai.data import Dataset, DataLoader
from monai.networks.nets import resnet50
from torchmetrics.classification import MultilabelAUROC, MultilabelF1Score, MultilabelAveragePrecision

from preprocessing_multiwindow import MultiWindowIntensityd

# Config - match the training run 
DATA_DIR = os.environ.get("CT_DATA_DIR", "./ct_rate_data/dataset/train_fixed")
LABEL_FILE = os.environ.get("CT_LABEL_FILE", "./ct_rate_data/dataset/multi_abnormality_labels/train_predicted_labels.csv")
SEED = int(os.environ.get("SEED", 12345))
total_scans = int(os.environ.get("TOTAL_SCANS", 1000))
max_volumes_per_patient = int(os.environ.get("MAX_VOLUMES_PER_PATIENT", 2))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", 4))
CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH", "resnet50_multiwindow_best.pth")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABEL_COLS = ["Medical material", "Arterial wall calcification", "Cardiomegaly", "Pericardial effusion",
              "Coronary artery wall calcification", "Hiatal hernia", "Lymphadenopathy", "Emphysema",
              "Atelectasis", "Lung nodule", "Lung opacity", "Pulmonary fibrotic sequela", "Pleural effusion",
              "Mosaic attenuation pattern", "Peribronchial thickening", "Consolidation", "Bronchiectasis",
              "Interlobular septal thickening"]


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


class CTRateDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        self.label_cols = LABEL_COLS

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sample = {"image": row['image_path'], "labels": row[self.label_cols].values.astype('float32')}
        return self.transform(sample) if self.transform else sample


eval_transforms = Compose([
    LoadImaged(keys=["image"]),
    EnsureChannelFirstd(keys=["image"]),
    Orientationd(keys=["image"], axcodes="RAS"),
    Spacingd(keys=["image"], pixdim=(2.0, 2.0, 2.0), mode="bilinear"),
    Resized(keys=["image"], spatial_size=(128, 128, 128)),
    MultiWindowIntensityd(keys=["image"]),
    EnsureTyped(keys=["image", "labels"])
])

test_ds = CTRateDataset(test_df, transform=eval_transforms)
test_loader = DataLoader(test_ds, batch_size=1, num_workers=NUM_WORKERS, shuffle=False)


def evaluate():
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Error: {CHECKPOINT_PATH} not found. Has training finished / saved a checkpoint yet?")
        return

    print(f"Loading saved model from {CHECKPOINT_PATH}...")
    model = resnet50(spatial_dims=3, n_input_channels=3, num_classes=18).to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()

    auroc_metric = MultilabelAUROC(num_labels=18, average="macro").to(device)
    f1_metric = MultilabelF1Score(num_labels=18, average="macro").to(device)
    pr_auc_metric = MultilabelAveragePrecision(num_labels=18, average="macro").to(device)

    auroc_per_class = MultilabelAUROC(num_labels=18, average=None).to(device)
    f1_per_class = MultilabelF1Score(num_labels=18, average=None).to(device)

    print("Starting evaluation scan...")
    with torch.no_grad():
        for batch_data in test_loader:
            inputs = batch_data["image"].to(device)
            labels = batch_data["labels"].to(device)

            outputs = model(inputs)
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
    print(f"Model 5 (multi-window 3D) | Final test results | "
          f"AUROC: {final_auroc:.4f} | PR-AUC: {final_prauc:.4f} | F1-Score: {final_f1:.4f}")

    print("\nPer-class breakdown:")
    print(f"{'Label':<40} {'AUROC':>8} {'F1':>8}")
    for label, a, f in zip(LABEL_COLS, auroc_by_class, f1_by_class):
        print(f"{label:<40} {a:>8.4f} {f:>8.4f}")


if __name__ == "__main__":
    evaluate()