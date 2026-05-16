import os
import torch
import pandas as pd
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
    CropForegroundd, ResizeWithPadOrCropd, RandFlipd, RandAffined,
    RandGaussianNoised, RandScaleIntensityd, RandShiftIntensityd,
    EnsureTyped, Resized, NormalizeIntensityd
)
from monai.data import Dataset, DataLoader
from monai.networks.nets import resnet50
from torchmetrics.classification import MultilabelAUROC, MultilabelF1Score, MultilabelAveragePrecision



class CTRateDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        print(f"Loading locked test labels from {csv_path}...")
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.label_cols = [
            "Medical material", "Arterial wall calcification", "Cardiomegaly", "Pericardial effusion",
            "Coronary artery wall calcification", "Hiatal hernia", "Lymphadenopathy", "Emphysema",
            "Atelectasis", "Lung nodule", "Lung opacity", "Pulmonary fibrotic sequela", "Pleural effusion",
            "Mosaic attenuation pattern", "Peribronchial thickening", "Consolidation", "Bronchiectasis",
            "Interlobular septal thickening"
        ]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Use 'image_path' which was saved by the training script
        sample = {
            "image": row['image_path'], 
            "labels": row[self.label_cols].values.astype('float32')
        }
        if self.transform:
            sample = self.transform(sample)
        return sample
    
# Evaluate
def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # same transforms as val
    eval_transforms = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(
            keys=["image"],
            pixdim=(2.0, 2.0, 2.0),
            mode="bilinear"
        ),
        Resized(
            keys=["image"],
            spatial_size=(128, 128, 128)
        ),
        NormalizeIntensityd(
            keys=["image"],
            nonzero=True
        ),
        EnsureTyped(keys=["image"])
    ])

    test_csv = 'test_patients_v2_locked.csv'

    if not os.path.exists(test_csv):
        print(f'Error: {test_csv} not found. Run train_baseline.py first to generate the test split.')
        return
    
    test_ds = CTRateDataset(csv_path = test_csv, transform=eval_transforms)
    test_loader = DataLoader(test_ds, batch_size=1, num_workers=4, shuffle=False)

    # load saved model
    print("Loading saved model...")
    model = resnet50(spatial_dims=3, n_input_channels=1, num_classes=18).to(device)
    model.load_state_dict(torch.load("resnet50_v2_best.pth", map_location=device))

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