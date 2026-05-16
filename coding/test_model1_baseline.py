import os
import torch
import pandas as pd
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
    Resized, ScaleIntensityRanged, EnsureTyped
)
from monai.data import Dataset, DataLoader
from monai.networks.nets import resnet50
from torchmetrics.classification import MultilabelAUROC, MultilabelF1Score, MultilabelAveragePrecision


# Reuse dataset logic from train model, but start from the bottom so we dont reuse the same (Data leakage)
class CTRateDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        print("Loading evaluation labels...")
        full_df = pd.read_csv(csv_file)
        
        # start from the bottom to avoid training data.
        full_df = full_df.iloc[::-1] 
        
        self.transform = transform
        self.label_cols = [
            "Medical material", "Arterial wall calcification", "Cardiomegaly", "Pericardial effusion",
            "Coronary artery wall calcification", "Hiatal hernia", "Lymphadenopathy", "Emphysema",
            "Atelectasis", "Lung nodule", "Lung opacity", "Pulmonary fibrotic sequela", "Pleural effusion",
            "Mosaic attenuation pattern", "Peribronchial thickening", "Consolidation", "Bronchiectasis",
            "Interlobular septal thickening"
        ]

        valid_records = []
        
        for idx, row in full_df.iterrows():
            raw_volume_name = row['VolumeName']
            base_name = raw_volume_name.replace('.nii.gz', '')  
            parts = base_name.split('_')
            folder_prefix = f"{parts[0]}_{parts[1]}"
            patient_folder = base_name.rsplit('_', 1)[0] 
            
            path_fixed = f"./ct_rate_data/dataset/train_fixed/{folder_prefix}/{patient_folder}/{raw_volume_name}"
            path_normal = f"./ct_rate_data/dataset/train/{folder_prefix}/{patient_folder}/{raw_volume_name}"
            
            # only downloaded train_fixed/train (not valid), we just check those
            if os.path.exists(path_fixed):
                row_dict = row.to_dict()
                row_dict['true_path'] = path_fixed
                valid_records.append(row_dict)
            elif os.path.exists(path_normal):
                row_dict = row.to_dict()
                row_dict['true_path'] = path_normal
                valid_records.append(row_dict)
                
            # Grab 100 unseen scans for testing
            if len(valid_records) >= 100: 
                break
                
        self.labels_df = pd.DataFrame(valid_records)
        print(f"Found {len(self.labels_df)} unseen scans for testing.")

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

# Evaluate
def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # same transforms as training
    transforms = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=(2.0, 2.0, 2.0), mode="bilinear"),
        Resized(keys=["image"], spatial_size=(128, 128, 128)),
        ScaleIntensityRanged(keys=["image"], a_min=-1000, a_max=400, b_min=0.0, b_max=1.0, clip=True),
        EnsureTyped(keys=["image"])
    ])

    # load data
    test_ds = CTRateDataset(csv_file="./ct_rate_data/dataset/multi_abnormality_labels/train_predicted_labels.csv", root_dir="./ct_rate_data", transform=transforms)
    test_loader = DataLoader(test_ds, batch_size=1, num_workers=0)

    # load saved model
    print("Loading saved model...")
    model = resnet50(spatial_dims=3, n_input_channels=1, num_classes=18).to(device)
    model.load_state_dict(torch.load("resnet50_baseline_best.pth", map_location=device))

    model.eval() # Set to evaluation mode

    # metrics for testing ("macro" ensures rare diseases matter equally)
    auroc_metric = MultilabelAUROC(num_labels=18, average="macro").to(device)
    f1_metric = MultilabelF1Score(num_labels=18, average="macro").to(device)
    pr_auc_metric = MultilabelAveragePrecision(num_labels=18, average="macro").to(device)


    print("Starting evaluation scan...")
    with torch.no_grad(): 
        for batch_data in test_loader:
            inputs = batch_data["image"].to(device)
            labels = batch_data["labels"].to(device)
            
            outputs = model(inputs)
            # apply sigmoid for probabilities
            probabilities = torch.sigmoid(outputs) 

            #strip MONAI metadata 
            clean_probs = probabilities.as_tensor()
            clean_labels = labels.long()
            
            auroc_metric.update(clean_probs, clean_labels)
            f1_metric.update(clean_probs, clean_labels)
            pr_auc_metric.update(clean_probs, clean_labels)

    # Compute final metric scores
    final_auroc = auroc_metric.compute().item()
    final_f1 = f1_metric.compute().item()
    final_prauc = pr_auc_metric.compute().item()

    print("\n" + "="*35)
    print(f"Final test results | AUROC: {final_auroc:.4f} | PR-AUC: {final_prauc:.4f} | F1-Score: {final_f1:.4f}")

if __name__ == "__main__":
    evaluate()