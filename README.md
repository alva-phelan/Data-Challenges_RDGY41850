# Data-Challenges_RDGY41850 — VLM3D Task 2: Multi-Label Abnormality Classification

Code and paper for a UCD RDGY41850 Data Challenges submission to the
[VLM3D Challenge](https://vlm3dchallenge.com/) Task 2 (18-class multi-label
thoracic abnormality classification on CT-RATE).

A 3D ResNet50 trained from scratch against a 2.5D classifier that
adapts a frozen/fine-tuned ImageNet-pretrained 2D ResNet50 via attention
pooling over sampled slices were compared, under a 2×2 design crossing architecture
(3D vs. 2.5D) with HU windowing (single-window vs. multi-window
lung/soft-tissue/bone). The full write-up is availible here.


## Models

| # | Script | Description |
|---|--------|-------------|
| 1 | `train_model1_baseline.py` | 3D ResNet50, random scan-level split (has data leakage — kept for comparison) |
| 2 | `train_model2_fixed_dataleakage.py` | 3D ResNet50, patient-level `GroupShuffleSplit`, single HU window |
| 3 | `train_model3_diff_windowing.py` | Model 2 + `NormalizeIntensityd` instead of fixed HU window |
| 4 | `train_model4_pretrained2d_fm.py` | 2.5D: shared ImageNet-pretrained ResNet50 backbone + attention pooling. Frozen or fine-tuned; single- or multi-window, via env vars below |
| 5 | `train_model5_multiwindow_3d.py` | Model 2 + multi-window (3-channel lung/soft-tissue/bone) input |

Each `train_model*.py` has a matching `test_model*.py` that reconstructs the
identical patient-level test split (same `SEED`) and reports macro-averaged
AUROC / F1 / PR-AUC, plus per-class breakdowns for Models 4 and 5.

## Setup

**Data.** Download [CT-RATE](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE)
and arrange it so that, relative to your working directory:
```
./ct_rate_data/dataset/train_fixed/<PatientFolder_Prefix>/<PatientFolder>/<VolumeName>.nii.gz
./ct_rate_data/dataset/multi_abnormality_labels/train_predicted_labels.csv
```
Paths can be overridden per-script via `CT_DATA_DIR` / `CT_LABEL_FILE`
environment variables (Models 4/5 and their test scripts; Models 1–3 have
these hardcoded at the top of the file).

**Environment.**
```bash
python -m venv .venv
source .venv/bin/activate        # or .venv\Scripts\activate on Windows
pip install -r requirements-cluster.txt --break-system-packages   # training/eval (GPU)
# or, for the notebook/visualisation side only:
pip install -r requirements.txt
```

**Checkpoints are not included in this repository.** Running the training
commands below will regenerate them locally. Running the test scripts
requires a checkpoint to already exist from the matching training run.
`requirements.txt` is what training/evaluation ran under on
the SLURM cluster. 

## Reproducing results

```bash
# Baseline iterations (Table 1)
python scripts/train_model1_baseline.py && python scripts/test_model1_baseline.py
python scripts/train_model2_fixed_dataleakage.py && python scripts/test_model2.py
python scripts/train_model3_diff_windowing.py && python scripts/test_model3.py

# 2x2 architecture x windowing design (Table 2) + fine-tuning ablation (Table 4)
python scripts/train_model5_multiwindow_3d.py && python scripts/test_model5_multiwindow_3d.py

FREEZE_BACKBONE=True  USE_MULTIWINDOW=False python scripts/train_model4_pretrained2d_fm.py
FREEZE_BACKBONE=True  USE_MULTIWINDOW=False python scripts/test_model4_pretrained2d_fm.py

FREEZE_BACKBONE=True  USE_MULTIWINDOW=True  python scripts/train_model4_pretrained2d_fm.py
FREEZE_BACKBONE=True  USE_MULTIWINDOW=True  python scripts/test_model4_pretrained2d_fm.py

FREEZE_BACKBONE=False USE_MULTIWINDOW=True  python scripts/train_model4_pretrained2d_fm.py
FREEZE_BACKBONE=False USE_MULTIWINDOW=True  python scripts/test_model4_pretrained2d_fm.py

# Dataset statistics + computational complexity (parameters, FLOPs, inference time)
pip install fvcore --break-system-packages
python scripts/generate_paper_stats.py
```

All test scripts for Models 4/5 **recompute the train/val/test split
independently** rather than reading a saved CSV. This only reproduces the
exact split used at training time if `SEED`, `TOTAL_SCANS`, and
`MAX_VOLUMES_PER_PATIENT` are unchanged from their defaults (`12345`, `1000`,
`2`). Don't change these when evaluating an existing checkpoint.


## Paper

Full write-up, including method details, the full 2×2 results table,
per-class breakdown, ablations, and explainability analysis (Grad-CAM,
attention-weight visualisation). LaTeX source and compiled PDF not yet
committed to this repository.

## Citation

If you use this code, please also cite the CT-RATE dataset:
```bibtex
@article{hamamci2024developing,
  title={Developing Generalist Foundation Models from a Multimodal Dataset for 3D Computed Tomography},
  author={Hamamci, Ibrahim Ethem and Er, Sezgin and Almas, Furkan and others},
  journal={arXiv preprint arXiv:2403.17834},
  year={2024}
}
```
