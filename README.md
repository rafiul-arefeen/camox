# CAMO-X: Concealment-Aware, Material-gated, Open-set X-ray threat recognition on STCray


## Environment Setup
```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install timm opencv-python pandas openpyxl scikit-learn scipy matplotlib tqdm pillow ipykernel ipywidgets
pip install ultralytics open_clip_torch
```

## Dataset
The STCray dataset is used introduced as a part of the paper [STING-BEE: Towards Vision-Language Model for Real-World X-ray Baggage Security Inspection](https://openaccess.thecvf.com/content/CVPR2025/papers/Velayudhan_STING-BEE_Towards_Vision-Language_Model_for_Real-World_X-ray_Baggage_Security_Inspection_CVPR_2025_paper.pdf)

[STCray-Dataset On Hugging Face](https://huggingface.co/datasets/Naoufel555/STCray-Dataset)

The files `STCray_TestSet.rar` and `STCray_TrainSet.rar` are needed.

## How to Run
The study is split into **named experiments** (`S1_main`, `A1_no_material`, `B2_convnext_nano`, ...). Each one
saves everything it produces under `WORK_DIR/runs/<name>/`.

1. **Every session:** run Part A (a minute or two once the data has been prepared).
2. **Choose today's work:** in cell A.3, list the experiments in `TODAY`, for example
   `TODAY = ["S1_main", "A0_ref", "A1_no_material"]`. Optionally set `SESSION_HOURS`.
3. **Run the notebook** (Run All is fine). Experiment cells whose name is not in `TODAY` only print their status.
4. **Next time:** change `TODAY` to the next experiments, e.g. `["A2_early_fusion", "A3_add_fusion"]`, and run again.

| Session | `TODAY` |
| --- | --- |
| 1 | `["S1_main", "A0_ref", "A1_no_material"]` | 
| 2 | `["B2_convnext_nano", "A11_no_tip", "A6_no_mask_head", "A7_gap_head"]` | 
| 3 | `["B1_resnet50", "A8_bce", "A9_focal", "A10_cb_bce"]` | 
| 4 | `["S2_zoom", "C1_fusion", "D1_clutter", "D1_clean", "B6_clip"]` | 
| 5 | `["A2_early_fusion", "A3_add_fusion", "A4_grayscale", "A5_hue_jitter", "A12_random_tip", "A13_384px"]` | 
| 6 | `["E1_open_s1", "E1_open_bank", "E1_eval", "C2_anomaly_proposals", "D1_both", "D2_random", "D2_4k", "D3_layer3"]` | 
| optional | `["S2_zoom_nomat", "C4_no_material", "B3_effnet_b4", "B4_vit_s16_384", "B5_yolov8s"]` | 
| optional | `["S1_main_seed1", "B2_convnext_nano_seed1", "S1_main_seed2", "B2_convnext_nano_seed2"]` | 

## The CAMOX Library
The project code lives in a small package next to this notebook (`camox/*.py`); the cell below imports it.
Real modules are needed on Windows because DataLoader workers are started with *spawn* and cannot load
classes defined inside a notebook.

| Module | Contents |
| --- | --- |
| `config` | class names, label synonyms, default hyper-parameters |
| `indexing` | folder scan, LabelMe (and other) annotation parser, captions |
| `splits` | perceptual signatures, near-duplicate groups, grouped multi-label split, leakage audit |
| `material` | the fixed material prior |
| `cache` | 512 px cache with soft masks, CA-TIP donors, Stage 2 zoom patches |
| `datasets` | Stage 1 dataset (CA-TIP, safe augmentations), Stage 2 crop dataset |
| `models` | material stream, gated fusion, FPN, CSRA head, mask head, plain timm baselines |
| `losses` | ASL, focal, class-balanced BCE, BCE, BCE + Dice |
| `engine` | AMP training loop, automatic micro-batch, accumulation, per-epoch checkpoints, inference |
| `evaluate` | mAP, F1, screening metrics, bootstrap CIs, Dice, pointing game, box AP |
| `anomaly` | PatchCore-style clutter bank and anomaly maps |
| `cascade` | proposals, crop aggregation, fusion |
| `baselines` | YOLOv8 and zero-shot CLIP |
| `viz` | plots |
| `synthetic` | tiny STCray look-alike for the smoke test |
| `pipeline` | experiment building blocks |
| `study` | **experiment registry and session runner** used by the cells below |