"""CAMO-X configuration: class names, label synonyms and experiment presets."""
from __future__ import annotations

import copy

# Index = STCray folder id - 1  (folder 21 = Multilabel Threat, folder 22 = Non Threat)
CLASS_NAMES = [
    "Explosive", "Gun", "3D Gun", "Knife", "Cutter", "Blade", "Shaving Razor",
    "Lighter", "Injection", "Battery", "Nail Cutter", "Other Sharp Item",
    "Powerbank", "Scissors", "Hammer", "Pliers", "Wrench", "Screwdriver",
    "Handcuffs", "Bullet",
]
NUM_CLASSES = len(CLASS_NAMES)
MULTI_FOLDER_ID = 21
BENIGN_FOLDER_ID = 22

# Normalised label string (lower-case, alphanumerics only) -> class index.
# Extend this dict if the diagnostics cell reports unmapped labels.
SYNONYMS = {
    "explosive": 0, "explosives": 0, "ied": 0, "ieds": 0, "bomb": 0,
    "improvisedexplosivedevice": 0, "explosivedevice": 0, "detonator": 0,
    "gun": 1, "guns": 1, "pistol": 1, "handgun": 1, "firearm": 1, "revolver": 1,
    "3dgun": 2, "3dguns": 2, "3dprintedgun": 2, "3dprintedfirearm": 2,
    "printedgun": 2, "gun3d": 2, "threedgun": 2, "3dpistol": 2,
    "knife": 3, "knives": 3,
    "cutter": 4, "cutters": 4, "boxcutter": 4, "papercutter": 4, "utilityknife": 4,
    "blade": 5, "blades": 5, "razorblade": 5,
    "shavingrazor": 6, "razor": 6, "shaver": 6,
    "lighter": 7, "lighters": 7,
    "injection": 8, "injections": 8, "syringe": 8, "syringes": 8, "needle": 8,
    "battery": 9, "batteries": 9,
    "nailcutter": 10, "nailcutters": 10, "nailclipper": 10, "nailclippers": 10,
    "othersharpitem": 11, "othersharpitems": 11, "sharpitem": 11,
    "othersharp": 11, "sharpobject": 11,
    "powerbank": 12, "powerbanks": 12,
    "scissors": 13, "scissor": 13,
    "hammer": 14, "hammers": 14,
    "pliers": 15, "plier": 15,
    "wrench": 16, "wrenches": 16, "spanner": 16,
    "screwdriver": 17, "screwdrivers": 17,
    "handcuffs": 18, "handcuff": 18, "cuffs": 18,
    "bullet": 19, "bullets": 19, "ammunition": 19, "ammo": 19, "cartridge": 19,
}
# Labels that mean "no threat" / generic placeholders.
IGNORE_LABELS = {"nonthreat", "benign", "normal", "background", "bg", "none", "negative", "safe"}
GENERIC_LABELS = {"threat", "object", "item", "prohibited", "prohibiteditem", "dangerous", "target", ""}

LOOKALIKE_GROUPS = {
    "Firearms": ["Gun", "3D Gun"],
    "Sharp": ["Knife", "Cutter", "Blade", "Shaving Razor", "Other Sharp Item", "Nail Cutter", "Scissors"],
    "Power": ["Battery", "Powerbank"],
    "Tools": ["Hammer", "Pliers", "Wrench", "Screwdriver"],
}

# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------
BASE_STAGE1 = dict(
    arch="camox",                 # 'camox' (custom) or 'simple' (plain timm classifier)
    stage=1,
    backbone="convnext_nano.r384_in12k_ft_in1k",
    pretrained=True,
    img_size=512,
    input_mode="rgb",             # 'rgb' | 'gray'
    material="gated",             # 'gated' | 'add' | 'early' | 'none'
    head="csra",                  # 'csra' | 'gap'
    csra_lambda=1.0,
    csra_temps=(1, 99),
    seg=True,
    lambda_seg=1.0,
    fpn_ch=128,
    loss="asl",                   # 'asl' | 'bce' | 'focal' | 'cb'
    asl_gamma_neg=4.0, asl_gamma_pos=0.0, asl_clip=0.05,
    focal_gamma=2.0, cb_beta=0.999,
    tip="concealed",              # 'concealed' | 'random' | 'none'
    tip_prob=0.5,
    hue_jitter=False,
    geo_aug=True,
    epochs=20,
    train_fraction=1.0,
    lr=1e-3,                      # new layers
    backbone_lr=2e-4,             # pretrained backbone
    weight_decay=0.05,
    warmup_epochs=1.0,
    eff_batch=32,
    batch_size="auto",           # int or 'auto' (largest micro-batch that fits)
    max_batch=32,
    grad_clip=5.0,
    amp=True,
    channels_last=True,
    num_workers=4,
    seed=0,
    exclude_classes=(),           # class names removed from training (open-set runs)
    eval_every=1,
    patience=None,                # early stopping patience in epochs (None = off)
    tta_flip=False,
)

BASE_STAGE2 = dict(
    BASE_STAGE1,
    stage=2,
    img_size=224,
    seg=False,
    lambda_seg=0.0,
    tip="none",
    epochs=15,
    eff_batch=64,
    max_batch=64,
    lr=1e-3,
    backbone_lr=2e-4,
    crop_context=1.3,             # test-time crop = context x box size
    train_context=(1.1, 1.8),
    train_jitter=0.15,
)

BASE_SIMPLE = dict(
    BASE_STAGE1,
    arch="simple",
    material="none",
    head="gap",
    seg=False,
    lambda_seg=0.0,
    loss="bce",
    tip="none",
)

BACKBONE_LR = {  # sensible fine-tuning learning rates per family
    "vit": 5e-5,
    "efficientnet": 5e-4,
    "resnet": 5e-4,
    "convnext": 2e-4,
}

PRESETS = {
    # full-budget runs (main model, baselines)
    "full": dict(epochs=20, train_fraction=1.0),
    # reduced budget shared by all ablations (same data subset + seed for fairness)
    "ablation": dict(epochs=12, train_fraction=0.5),
    # a few minutes, just to prove the whole notebook runs
    "smoke": dict(epochs=1, train_fraction=1.0, img_size=128, max_batch=8, eff_batch=8,
                  batch_size=8, num_workers=0, pretrained=False),
}


def make_cfg(base: dict, **overrides) -> dict:
    cfg = copy.deepcopy(base)
    for k, v in overrides.items():
        if k not in cfg and k not in ("name", "group", "notes", "preset"):
            raise KeyError(f"Unknown config key: {k}")
        cfg[k] = v
    return cfg


def apply_preset(cfg: dict, preset: str, smoke: bool = False) -> dict:
    cfg = copy.deepcopy(cfg)
    cfg.update(PRESETS[preset])
    if smoke:
        stage2 = cfg.get("stage") == 2
        smoke_over = dict(PRESETS["smoke"])
        if stage2:
            smoke_over["img_size"] = 96
        cfg.update(smoke_over)
    cfg["preset"] = preset
    return cfg


def backbone_lr_for(backbone: str) -> float:
    b = backbone.lower()
    for key, lr in BACKBONE_LR.items():
        if key in b:
            return lr
    return 2e-4
