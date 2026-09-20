"""Session-friendly study runner.

Every experiment (main model, each baseline, each ablation, ...) has a name, a config, dependencies and a state
that lives on disk under WORK_DIR/runs/<name>. You can run any subset in any session; finished work is never
redone, interrupted training resumes from its last epoch, and the result tables grow as runs finish.
"""
from __future__ import annotations

import importlib
import json
import shutil
import time
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from . import anomaly as AN
from . import baselines as BL
from . import cache as CA
from . import cascade as CS
from . import datasets as DS
from . import engine as EN
from . import evaluate as EV
from . import indexing as IX
from . import material as MA
from . import pipeline as PL
from . import splits as SP
from . import viz as VZ
from .config import (BASE_SIMPLE, BASE_STAGE1, BASE_STAGE2, CLASS_NAMES, LOOKALIKE_GROUPS, NUM_CLASSES,
                     backbone_lr_for, make_cfg)
from .utils import crop_with_pad, fmt_time, get_device, imread_rgb, load_json, save_json

try:
    from IPython.display import Markdown, display
except Exception:  # pragma: no cover
    def display(x):
        print(x)

    def Markdown(x):
        return x

# keys that do not change what a run learns (safe to differ between sessions)
IGNORE_KEYS = {"num_workers", "batch_size", "max_batch", "eval_every", "preset", "amp", "channels_last",
               "tta_flip", "patience", "name", "group", "notes"}
RESULT_TABLE = {"main": "stage1", "baseline": "stage1", "ablation": "ablations", "anomaly": "anomaly",
                "cascade": "cascade", "openset": "openset_known", "seed": "seeds"}
PIP_NAME = {"ultralytics": "ultralytics", "open_clip": "open_clip_torch"}
# extras that are not needed for the core story (the seed repeats are optional as well)
OPTIONAL = ("S2_zoom_nomat", "C4_no_material", "B3_effnet_b4", "B4_vit_s16_384", "B5_yolov8s")
# result rows written by experiments whose row names differ from the experiment name
DERIVED_ROWS = {
    "C1_fusion": {"cascade": ("C1_mask_", "C2_oracle_", "C3_top", "C3_ctx"),
                  "box_ap": ("stage1_only", "cascade", "zoom_on_gt"), "proposals": ("mask",)},
    "C2_anomaly_proposals": {"cascade": ("C2_mask_anom_",), "proposals": ("mask_anom",)},
    "C4_no_material": {"cascade": ("C4_s2_nomat_",)},
    "B5_yolov8s": {"box_ap": ("yolo",)},
}


def _show(df, digits=4):
    if df is None or len(df) == 0:
        return
    d = df.copy()
    num = d.select_dtypes("number").columns
    d[num] = d[num].round(digits)
    display(d)


def _norm_cfg(cfg):
    return json.loads(json.dumps({k: v for k, v in cfg.items() if k not in IGNORE_KEYS}, default=str))


# =================================================================================== data preparation
def prepare_data(ctx, data_root=None, train_dir=None, test_dir=None, val_frac=0.15, dup_bits=12, workers=8,
                 bbox_format="auto", log=print):
    """Index -> 512 px cache -> leakage-aware split -> TIP donors. Every step is cached in WORK_DIR, so after the
    first session this only loads a few files."""
    w = ctx.work
    f_index, f_diag = w / "index.pkl", w / "index_diag.pkl"
    f_cache, f_split, f_donor = w / f"index_cached_{ctx.cache_size}.pkl", w / "index_split.pkl", w / "donors.pkl"
    t0 = time.time()
    if f_split.exists():
        df = pd.read_pickle(f_split)
    else:
        if f_cache.exists():
            df = pd.read_pickle(f_cache)
        else:
            if f_index.exists():
                df = pd.read_pickle(f_index)
            else:
                log("Step 1/4: indexing the dataset (one time only) ...")
                df, diag = IX.build_index(data_root, train_dir, test_dir, workers=workers, bbox_format=bbox_format)
                df.to_pickle(f_index)
                pd.to_pickle(diag, f_diag)
            diag = pd.read_pickle(f_diag)
            log("Step 2/4: caching every scan at 512 px (one time only, the longest step) ...")
            df = CA.build_cache(df, ctx.cache, ctx.cache_size, workers=workers, bbox_format=diag["bbox_format"])
            df.to_pickle(f_cache)
            CA.cache_progress_file(ctx.cache, ctx.cache_size).unlink(missing_ok=True)
        df["cache_ok"] = df["cache_ok"].fillna(False).astype(bool)
        log("Step 3/4: near-duplicate groups and the train/validation split (one time only) ...")
        ok = df["cache_ok"].to_numpy()
        tr = (df["split"] == "train").to_numpy() & ok
        te = (df["split"] == "test").to_numpy() & ok
        sig_tr = df.loc[tr, ["sig_d", "sig_a"]].to_numpy()
        groups, used_bits = SP.group_near_duplicates(sig_tr, max_dist=dup_bits)
        Ytr = IX.label_matrix(df[tr])
        val = SP.grouped_multilabel_split(Ytr, groups, val_frac, seed=0,
                                          extra_strata=(Ytr.sum(1, keepdims=True) == 0).astype(np.float32))
        df["part"] = np.where(df["split"] == "test", "test", "train")
        idx = np.flatnonzero(tr)
        df.loc[df.index[idx[val]], "part"] = "val"
        df["dup_group"] = -1
        df.loc[df.index[idx], "dup_group"] = groups
        _, dup = SP.leakage_audit(sig_tr, df.loc[te, ["sig_d", "sig_a"]].to_numpy(), used_bits)
        df["near_dup_of_train"] = False
        df.loc[df.index[np.flatnonzero(te)], "near_dup_of_train"] = dup
        df.attrs["dup_bits"] = used_bits
        df.to_pickle(f_split)
    diag = pd.read_pickle(f_diag) if f_diag.exists() else None
    ctx.df = df
    if f_donor.exists():
        donors = pd.read_pickle(f_donor)
    else:
        log("Step 4/4: cutting CA-TIP threat donors from training scans (one time only) ...")
        donors = CA.build_donors(PL.part_df(ctx, "train"), ctx.cache, ctx.cache_size, max_per_class=1500,
                                 workers=workers)
        donors.to_pickle(f_donor)
    ctx.donors = donors
    parts = df[df["cache_ok"]]["part"].value_counts().to_dict()
    log(f"data ready in {fmt_time(time.time() - t0)}: {parts} scans, {len(donors)} TIP donors")
    return df, diag, donors


# =================================================================================== one-time audit / EDA
def annotation_audit(df, diag):
    if diag is None:
        print("no index diagnostics found")
        return
    print("Json_BB file style:", dict(diag.get("json_styles", {})))
    st = diag.get("shape_types", {})
    print("LabelMe shape types  Json_BB:", dict(st.get("Json_BB", {})), "| Json:", dict(st.get("Json", {})))
    print("Box format:", diag["bbox_format"], "-", diag["bbox_format_reason"])
    print("JSON files that failed to parse: Json_BB =", diag["bb_parse_errors"], "| Json =", diag["json_parse_errors"])
    print("Scans with issues:", dict(diag["issues"]))
    if diag["conflicts"]:
        print("Folder class vs. label disagreements (folder class kept):", diag["conflicts"].most_common(10))
    vt = IX.vocab_table(diag["label_vocab"])
    display(vt.head(40))
    bad = vt[vt["status"].isin(["unmapped", "generic", "missing"])]
    if len(bad):
        print("WARNING - labels that could not be mapped to a class (add real class names to SYNONYMS in config):")
        display(bad.head(30))
    ex = df[df.folder_id == 21].iloc[0] if (df.folder_id == 21).any() else df[df.bb_path.notna()].iloc[0]
    print("\n--- one Json_BB file:", ex.bb_path)
    print(IX.describe_json(ex.bb_path)[:1800] if ex.bb_path else "none")
    if ex.json_path:
        print("\n--- the matching Json (mask) file:", ex.json_path)
        print(IX.describe_json(ex.json_path)[:1000])
    print("\nParsed labels:", [CLASS_NAMES[c] for c in ex.labels])
    print("Parsed boxes (x1, y1, x2, y2, class):", ex.boxes[:5])
    if "box_mask_agree" in df and df["box_mask_agree"].notna().any():
        agree = df["box_mask_agree"].dropna()
        print(f"\nmask pixels inside the parsed boxes: median {agree.median():.2f} "
              f"({int((agree < 0.5).sum())} of {len(agree)} scans below 0.5)")
        if agree.median() < 0.6:
            print("WARNING: boxes and masks disagree - check the sample grid in the EDA cell.")


def eda(ctx, df):
    Y = IX.label_matrix(df)
    summary = pd.DataFrame({
        "scans": df.groupby("split").size(),
        "threat": df.groupby("split")["is_threat"].sum(),
        "multi-threat": df.assign(m=df.n_labels > 1).groupby("split")["m"].sum(),
        "clean": df.assign(c=~df.is_threat).groupby("split")["c"].sum(),
    })
    summary["clean %"] = (100 * summary["clean"] / summary["scans"]).round(2)
    display(summary)
    counts = {s: Y[(df.split == s).to_numpy()].sum(0) for s in ("train", "test")}
    cc = pd.DataFrame(counts, index=CLASS_NAMES).astype(int)
    cc["test/train"] = (cc["test"] / cc["train"].clip(lower=1)).round(2)
    display(cc.T)
    VZ.plot_class_counts(counts, path=ctx.fig("eda_class_counts.png"))
    plt.show()
    print(df[["img_w", "img_h"]].describe().T.round(1))
    rel = [(b[2] - b[0]) * (b[3] - b[1]) / (r.img_w * r.img_h) for r in df.itertuples() if r.img_w for b in r.boxes]
    VZ.plot_hist({"threat box": rel}, log=True, xlabel="box area / scan area",
                 title="Threat sizes (log scale)", path=ctx.fig("eda_box_sizes.png"))
    plt.show()
    print("objects per threat scan:", df[df.is_threat].n_boxes.describe().round(2).to_dict())
    cap = df["caption"].dropna().astype(str)
    print(f"{len(cap)}/{len(df)} scans have a caption. Examples:")
    for c in cap.sample(min(4, len(cap)), random_state=0):
        print(" -", c[:300])
    words = ["conceal", "hidden", "hide", "beneath", "under", "behind", "among", "inside", "overlap", "occlu",
             "cover", "wrap", "clutter"]
    if len(cap):
        display(pd.Series({w: round(float(cap.str.lower().str.contains(w).mean()), 3) for w in words},
                          name="share of captions mentioning").to_frame().T)


def split_summary(ctx):
    df = ctx.df
    Y = IX.label_matrix(df)
    masks = {p: ((df["part"] == p) & df["cache_ok"]).to_numpy() for p in ("train", "val", "test")}
    display(SP.split_report(df, Y, masks))
    gs = df[df.dup_group >= 0].groupby("dup_group").size()
    print(f"near-duplicate groups in the official train set: {len(gs)} (largest {gs.max()} scans, "
          f"{int((gs > 1).sum())} groups with more than one scan); threshold {df.attrs.get('dup_bits')} bits")
    n_test = int((df.part == "test").sum())
    print(f"test scans with a near-duplicate in train: {int(df.near_dup_of_train.sum())} of {n_test}")


def material_check(ctx):
    tr = PL.part_df(ctx, "train")
    sample = tr.sample(min(200, len(tr)), random_state=0)
    imgs = [CA.load_cached(u, ctx.cache, ctx.cache_size)[0] for u in sample["uid"]]
    centers, hist = MA.hue_histogram([i for i in imgs if i is not None])
    VZ.plot_hue_hist(centers, hist, MA.MATERIAL_HUES, path=ctx.fig("material_hue_histogram.png"))
    plt.show()
    print("most common hues (deg):", np.round(centers[np.argsort(-hist)[:6]]).astype(int).tolist())
    rows = []
    for cname in ["3D Gun", "Gun", "Explosive", "Knife", "Battery", "Powerbank", "Scissors", "Bullet"]:
        sub = tr[tr.labels.apply(lambda l: CLASS_NAMES.index(cname) in l)]
        if len(sub):
            rows.append(sub.iloc[0].to_dict())
    VZ.show_samples(rows, lambda u: CA.load_cached(u, ctx.cache, ctx.cache_size), path=ctx.fig("eda_samples.png"))
    plt.show()
    for r in rows[:2]:
        VZ.show_material(CA.load_cached(r["uid"], ctx.cache, ctx.cache_size)[0],
                         path=ctx.fig(f"material_{r['uid']}.png"))
        plt.show()


def tip_preview(ctx, img_size):
    display(ctx.donors["cls"].map(lambda c: CLASS_NAMES[c]).value_counts().to_frame("donors").T)
    ds = DS.Stage1Dataset(PL.items_of(PL.part_df(ctx, "train")), ctx.cache, ctx.cache_size, img_size, train=True,
                          tip="concealed", tip_prob=1.0, donors=ctx.donors)
    VZ.show_augmented(ds, list(range(min(6, len(ds)))), path=ctx.fig("tip_preview.png"))
    plt.show()


# =================================================================================== experiment registry
@dataclass
class Spec:
    name: str
    kind: str
    group: str
    label: str
    cfg: dict | None = None
    kw: dict = field(default_factory=dict)
    deps: tuple = ()
    est: float = 30.0          # rough minutes on an RTX 4060 (training + evaluation)
    needs: str | None = None   # optional python package


class Study:
    """Registry of every experiment + a dependency-aware, resumable, time-boxed runner."""

    def __init__(self, ctx, settings):
        self.ctx = ctx
        self.S = dict(settings)
        self.specs = {}
        self.deadline = None
        self._define()

    # ------------------------------------------------------------------ configs
    def _smoke(self, cfg, small):
        if self.S["smoke"]:
            cfg.update(img_size=small, batch_size=8, max_batch=8, eff_batch=8, pretrained=False, num_workers=0)
        return cfg

    def s1(self, budget, **over):
        cfg = make_cfg(BASE_STAGE1, **over)
        cfg.update(budget)
        cfg["num_workers"] = self.S["num_workers"]
        return self._smoke(cfg, 96 if over.get("img_size", 512) < 512 else 128)

    def sb(self, backbone, budget, **over):
        cfg = make_cfg(BASE_SIMPLE, backbone=backbone, backbone_lr=backbone_lr_for(backbone), **over)
        cfg.update(budget)
        cfg["num_workers"] = self.S["num_workers"]
        return self._smoke(cfg, 96 if "vit" in backbone else 128)

    def s2(self, **over):
        cfg = make_cfg(BASE_STAGE2, **over)
        cfg.update(epochs=self.S["s2_epochs"], num_workers=self.S["num_workers"])
        return self._smoke(cfg, 96)

    def _add(self, name, kind, group, label, cfg=None, deps=(), est=30.0, needs=None, **kw):
        self.specs[name] = Spec(name, kind, group, label, cfg, kw, tuple(deps), est, needs)

    def _define(self):
        S, a = self.S, self._add
        full, abl = S["full"], S["abl"]
        a("S1_main", "stage1", "main", "CAMO-X Stage 1 (Look), full budget", self.s1(full), est=150)
        a("B1_resnet50", "stage1", "baseline", "B1 ResNet-50 (GAP, BCE)", self.sb("resnet50.tv2_in1k", full), est=150)
        a("B2_convnext_nano", "stage1", "baseline", "B2 ConvNeXt-Nano (GAP, BCE)",
          self.sb("convnext_nano.r384_in12k_ft_in1k", full), est=120)
        a("B3_effnet_b4", "stage1", "baseline", "B3 EfficientNet-B4 (GAP, BCE)",
          self.sb("efficientnet_b4.ra2_in1k", full), est=210)
        a("B4_vit_s16_384", "stage1", "baseline", "B4 ViT-S/16 @384",
          self.sb("vit_small_patch16_384.augreg_in21k_ft_in1k", full, img_size=384), est=150)
        a("B5_yolov8s", "yolo", "baseline", "B5 YOLOv8s detector", est=150, needs="ultralytics")
        a("B6_clip", "clip", "baseline", "B6 zero-shot CLIP ViT-B/16", est=6, needs="open_clip")
        ablations = [
            ("A0_ref", {}, "A0 full Stage 1 (reference, reduced budget)"),
            ("A1_no_material", dict(material="none"), "A1 no material stream"),
            ("A2_early_fusion", dict(material="early"), "A2 material as 9-channel input"),
            ("A3_add_fusion", dict(material="add"), "A3 plain addition, no gate"),
            ("A4_grayscale", dict(input_mode="gray", material="none"), "A4 greyscale input"),
            ("A5_hue_jitter", dict(hue_jitter=True), "A5 + hue/saturation jitter"),
            ("A6_no_mask_head", dict(seg=False, lambda_seg=0.0), "A6 no mask head"),
            ("A7_gap_head", dict(head="gap"), "A7 average pooling instead of CSRA"),
            ("A8_bce", dict(loss="bce"), "A8 BCE instead of ASL"),
            ("A9_focal", dict(loss="focal"), "A9 focal loss"),
            ("A10_cb_bce", dict(loss="cb"), "A10 class-balanced BCE"),
            ("A11_no_tip", dict(tip="none"), "A11 no CA-TIP"),
            ("A12_random_tip", dict(tip="random"), "A12 TIP with random placement"),
            ("A13_384px", dict(img_size=384), "A13 384 px input"),
        ]
        for n, o, l in ablations:
            a(n, "stage1", "ablation", l, self.s1(abl, **o), est=20 if n == "A13_384px" else 32)
        bs = S["bank_size"]
        banks = [
            ("D1_clutter", dict(mode="clutter"), "D1 clutter bank (default)", 12),
            ("D1_clean", dict(mode="clean"), "B7 / D1 clean-bag bank", 5),
            ("D1_both", dict(mode="both"), "D1 clean + clutter bank", 12),
            ("D2_random", dict(mode="clutter", coreset=False), "D2 random subsample instead of coreset", 10),
            ("D2_4k", dict(mode="clutter", bank_size=max(100, bs // 4)), "D2 bank of 4k patches", 10),
            ("D3_layer3", dict(mode="clutter", layers="l3"), "D3 layer 3 features only", 12),
        ]
        for n, kw, l, e in banks:
            a(n, "bank", "anomaly", l, est=e, **dict(dict(bank_size=bs), **kw))
        a("S2_zoom", "stage2", "cascade", "Stage 2 (Zoom) classifier", self.s2(), deps=("S1_main",), est=60)
        a("S2_zoom_nomat", "stage2", "cascade", "Stage 2 without the material stream", self.s2(material="none"),
          deps=("S1_main",), est=40)
        a("C1_fusion", "cascade1", "cascade", "C1/C3 cascade fusion, oracle and box AP",
          deps=("S1_main", "S2_zoom"), est=25)
        a("C2_anomaly_proposals", "cascade2", "cascade", "C2 mask + anomaly-map proposals",
          deps=("S1_main", "S2_zoom", "D1_clutter"), est=15)
        a("C4_no_material", "cascade4", "cascade", "C4 Stage 2 without material",
          deps=("S1_main", "S2_zoom_nomat"), est=5)
        heldout = tuple(S["heldout"])
        a("E1_open_s1", "stage1", "openset", f"E1 Stage 1 trained without {', '.join(heldout)}",
          self.s1(abl, exclude_classes=heldout), est=30)
        a("E1_open_bank", "bank", "openset", "E1 clutter bank without the held-out classes", est=12,
          mode="clutter", bank_size=bs, exclude=heldout)
        a("E1_eval", "openset", "openset", "E1 unseen-threat evaluation", deps=("E1_open_s1", "E1_open_bank"), est=3)
        for seed in (1, 2):
            a(f"S1_main_seed{seed}", "stage1", "seed", f"CAMO-X Stage 1, seed {seed}", self.s1(full, seed=seed),
              est=150)
            a(f"B2_convnext_nano_seed{seed}", "stage1", "seed", f"B2 ConvNeXt-Nano, seed {seed}",
              self.sb("convnext_nano.r384_in12k_ft_in1k", full, seed=seed), est=120)

    def names(self, group=None):
        return [n for n, s in self.specs.items() if group is None or s.group == group]

    # ------------------------------------------------------------------ state on disk
    def run_dir(self, name):
        return self.ctx.runs / name

    def _table(self, name):
        f = self.ctx.results / f"{name}.csv"
        return pd.read_csv(f) if f.exists() else None

    def _evaluated(self, name):
        t = self._table(RESULT_TABLE[self.specs[name].group] if self.specs[name].kind != "bank" else "anomaly")
        return t is not None and name in set(t["run"])

    def _package_ok(self, spec):
        if not spec.needs:
            return True
        try:
            importlib.import_module(spec.needs)
            return True
        except ImportError:
            return False

    def state(self, name):
        sp, d = self.specs[name], self.run_dir(name)
        if sp.kind in ("stage1", "stage2"):
            if (d / "done.json").exists():
                if sp.kind == "stage1" and not self._evaluated(name):
                    return "trained, not evaluated"
                return "done"
            if (d / "progress.json").exists():
                p = load_json(d / "progress.json")
                return f"partial ({p['epoch']}/{p['epochs']} epochs)"
            return "not started"
        if sp.kind == "bank":
            if (d / "null_fit.json").exists() and self._evaluated(name):
                return "done"
            return "partial (bank built)" if (d / "bank.pt").exists() else "not started"
        if sp.kind == "yolo":
            if (d / "pred.pkl").exists() and self._evaluated(name):
                return "done"
            if (d / "done.json").exists():
                return "trained, not evaluated"
            return "partial" if (d / "weights" / "last.pt").exists() else "not started"
        if sp.kind == "clip":
            return "done" if (d / "pred.npz").exists() and self._evaluated(name) else "not started"
        return "done" if (d / "eval_done.json").exists() else "not started"

    def _speed(self, sp):
        if self.S["smoke"]:
            return 0.02
        if sp.kind not in ("stage1", "stage2"):
            return 1.0
        ratios = []
        for n, s in self.specs.items():
            f = self.run_dir(n) / "done.json"
            if s.kind == "stage1" and f.exists():
                mins = load_json(f).get("train_minutes")
                if mins:
                    ratios.append(mins / max(1.0, s.est - 8))
        return float(np.clip(np.median(ratios), 0.3, 3.0)) if ratios else 1.0

    def estimate(self, name, _seen=None):
        """Minutes still needed for `name`, including unfinished dependencies (rough)."""
        _seen = set() if _seen is None else _seen
        if name in _seen:
            return 0.0
        _seen.add(name)
        sp = self.specs[name]
        st = self.state(name)
        if st == "done":
            return 0.0
        own = sp.est * self._speed(sp)
        if st.startswith("partial") and (self.run_dir(name) / "progress.json").exists():
            p = load_json(self.run_dir(name) / "progress.json")
            own *= max(0.1, 1 - p["epoch"] / max(1, p["epochs"]))
        elif st.startswith("trained"):
            own = min(own, 8.0)
        return own + sum(self.estimate(d, _seen) for d in sp.deps if self.state(d) != "done")

    def _metric(self, name):
        sp = self.specs[name]
        try:
            if sp.kind == "bank":
                t = self._table("anomaly")
                r = t[t.run == name].iloc[0]
                return f"bag AUROC {r['bag_auroc_max']:.3f}"
            if sp.kind == "cascade1":
                t = self._table("cascade")
                r = t[t.run == "C1_mask_avg"].iloc[0]
                return f"cascade mAP {r['mAP']:.3f}"
            if sp.kind == "cascade2":
                t = self._table("cascade")
                return f"mAP {t[t.run == 'C2_mask_anom_avg'].iloc[0]['mAP']:.3f}"
            if sp.kind == "cascade4":
                t = self._table("cascade")
                return f"mAP {t[t.run == 'C4_s2_nomat_avg'].iloc[0]['mAP']:.3f}"
            if sp.kind == "openset":
                t = self._table("openset")
                r = t[(t.score == "fused (max z)") & (t.population == "unseen threats")].iloc[0]
                return f"unseen AUROC {r['auroc']:.3f}"
            if sp.kind == "stage2":
                f = self.run_dir(name) / "done.json"
                return f"crop val mAP {load_json(f)['best_val_map']:.3f}" if f.exists() else ""
            t = self._table(RESULT_TABLE[sp.group])
            r = t[t.run == name].iloc[0]
            txt = f"test mAP {r['mAP']:.3f}"
            if sp.group == "ablation":
                dlt = self._table("ablation_deltas")
                if dlt is not None and name in set(dlt.run):
                    txt += f" ({dlt[dlt.run == name].iloc[0]['delta']:+.3f} vs A0)"
            return txt
        except Exception:
            return ""

    def status(self, group=None, show=True):
        rows = []
        for n, sp in self.specs.items():
            if group and sp.group != group:
                continue
            st = self.state(n)
            f = self.run_dir(n) / "done.json"
            mins = np.nan
            if f.exists():
                try:
                    mins = round(float(load_json(f).get("train_minutes") or 0), 1)
                except Exception:
                    mins = np.nan
            if sp.needs and not self._package_ok(sp) and st != "done":
                st = f"{st} (needs: pip install {PIP_NAME.get(sp.needs, sp.needs)})"
            rows.append(dict(experiment=n, group=sp.group, what=sp.label, state=st, result=self._metric(n),
                             train_min=mins, est_min_left=0 if st == "done" else round(self.estimate(n)),
                             note="settings changed since training" if self.config_diff(n) else ""))
        df = pd.DataFrame(rows)
        if show:
            done = (df.state == "done").sum()
            left = df.loc[df.state != "done", "est_min_left"]
            core = df[(df.state != "done") & ~df.group.isin(["seed"]) & ~df.experiment.isin(OPTIONAL)]
            print(f"{done}/{len(df)} experiments done. Rough time left: {core.est_min_left.sum() / 60:.1f} h for the "
                  f"core study, {left.sum() / 60:.1f} h including optional baselines and seeds.")
            display(df)
        return df

    # ------------------------------------------------------------------ running
    def set_session(self, hours=None):
        self.deadline = time.time() + hours * 3600 if hours else None
        if hours:
            print(f"session limit: {hours} h - runs that would not finish in the remaining time are postponed")

    def config_diff(self, name):
        """Settings that differ between the saved run and the current notebook settings."""
        sp = self.specs[name]
        f = self.run_dir(name) / "config.json"
        if sp.cfg is None or not f.exists():
            return []
        old, new = _norm_cfg(load_json(f)), _norm_cfg(sp.cfg)
        return [f"{k}: {old.get(k)} -> {new.get(k)}" for k in sorted(set(old) | set(new)) if old.get(k) != new.get(k)]

    def _config_ok(self, name):
        diff = self.config_diff(name)
        if not diff:
            return True
        print(f"[{name}] WARNING: this run was started with different settings ({'; '.join(diff)}).")
        print(f"[{name}] Keeping the saved run. Restore the old settings, or call "
              f"study.reset('{name}') to delete it and train again.")
        return False

    def reset(self, name):
        """Delete everything saved for one experiment (checkpoints, predictions, result rows)."""
        d = self.run_dir(name)
        if d.exists():
            shutil.rmtree(d)
        derived = DERIVED_ROWS.get(name, {})
        for f in self.ctx.results.glob("*.csv"):
            t = pd.read_csv(f)
            if "run" not in t:
                continue
            runs = t["run"].astype(str)
            drop = runs == name
            for prefix in derived.get(f.stem, ()):
                drop |= runs.str.startswith(prefix)
            if name == "A0_ref" and f.stem == "ablation_deltas":
                drop[:] = True
            if drop.any():
                t[~drop].to_csv(f, index=False)
        if name == "C1_fusion":
            print("note: cached proposal patches are kept; delete WORK_DIR/cache/evalpatches_* to rebuild them")
        print(f"[{name}] reset - it will run from scratch next time")

    def cell(self, name, plan):
        """Run `name` if it is in this session's plan, otherwise just report its state."""
        todo = plan == "all" or (not isinstance(plan, str) and name in plan)
        if todo:
            return self.run(name)
        print(f"[{name}] {self.state(name)} - not in TODAY's plan")
        return None

    def run(self, name, _depth=0):
        if name not in self.specs:
            raise KeyError(f"unknown experiment '{name}'. Known: {', '.join(self.specs)}")
        sp = self.specs[name]
        st = self.state(name)
        if st == "done":
            print(f"[{name}] already done ({self._metric(name)}) - nothing to do")
            diff = self.config_diff(name)
            if diff:
                print(f"[{name}] note: it was trained with different settings ({'; '.join(diff)}); "
                      f"study.reset('{name}') retrains it with the current ones")
            return "done"
        if not self._package_ok(sp):
            print(f"[{name}] skipped: the optional package '{sp.needs}' is not installed")
            return None
        for dep in sp.deps:
            if self.state(dep) != "done":
                print(f"[{name}] needs '{dep}' first")
                self.run(dep, _depth + 1)
                if self.state(dep) != "done":
                    print(f"[{name}] postponed: '{dep}' is not finished yet")
                    return None
        if not self._config_ok(name):
            return None
        est = self.estimate(name)
        if self.deadline is not None:
            left = (self.deadline - time.time()) / 60
            if est > left:
                print(f"[{name}] postponed: needs about {est:.0f} min, {max(0.0, left):.0f} min left in this session")
                return None
        print(f"[{name}] {st} -> running (about {est:.0f} min): {sp.label}")
        t = time.time()
        getattr(self, f"_run_{sp.kind}")(sp)
        print(f"[{name}] finished in {fmt_time(time.time() - t)} - {self._metric(name)}")
        return "done"

    # ------------------------------------------------------------------ runners
    def _run_stage1(self, sp):
        PL.train_stage1(self.ctx, sp.name, sp.cfg)
        row = PL.evaluate_stage1(self.ctx, sp.name, sp.cfg, label=sp.label,
                                 latency=sp.group in ("main", "baseline"))
        PL.upsert_results(self.ctx, [row], RESULT_TABLE[sp.group])
        if sp.group == "ablation":
            self.update_deltas([sp.name] if sp.name != "A0_ref" else None)
        _show(pd.DataFrame([row])[[c for c in ("label", "mAP", "mAP_lo", "mAP_hi", "val_mAP", "macro_f1",
                                                 "bag_auroc", "mask_dice", "train_min") if c in row]])

    def update_deltas(self, names=None):
        """Paired-bootstrap mAP change vs A0 for finished ablations (all of them if names is None)."""
        if self.state("A0_ref") != "done":
            print("ablation deltas need A0_ref - run it in some session to get the comparison")
            return None
        names = names or [n for n in self.names("ablation") if n != "A0_ref" and self.state(n) == "done"]
        runs = [(n, self.specs[n].cfg, self.specs[n].label) for n in names]
        if not runs:
            return None
        d = PL.paired_deltas(self.ctx, "A0_ref", self.specs["A0_ref"].cfg, runs)
        if len(d):
            PL.upsert_results(self.ctx, d.to_dict("records"), "ablation_deltas")
        return d

    def _run_bank(self, sp):
        kw = sp.kw
        exclude = SP.class_ids(kw.get("exclude", ()))
        layers = kw.get("layers", "l23")
        PL.build_bank(self.ctx, sp.name, mode=kw["mode"], layers=layers, bank_size=kw["bank_size"],
                      coreset=kw.get("coreset", True), exclude_ids=exclude)
        row = PL.evaluate_bank(self.ctx, sp.name, sp.label, layers=layers)
        PL.upsert_results(self.ctx, [row], "anomaly")
        _show(pd.DataFrame([row]))

    def _cfg(self, name):
        return self.specs[name].cfg

    def proposal_threshold(self):
        f = self.run_dir("S1_main") / "proposal_threshold.json"
        if f.exists():
            return float(load_json(f)["thr"])
        thr, sweep = PL.sweep_proposal_thr(self.ctx, "S1_main", self._cfg("S1_main"), "val",
                                           topk=self.S["prop_topk"], max_per_scan=self.S["prop_max"])
        save_json(dict(thr=thr, sweep=sweep.to_dict("records")), f)
        print(f"mask threshold for proposals (picked on validation): {thr}")
        _show(sweep, 3)
        return thr

    def _patches(self):
        thr = self.proposal_threshold()
        return PL.build_stage2_patches(self.ctx, "S1_main", self._cfg("S1_main"), tag=f"main_t{thr:.2f}",
                                       thr=thr, topk=self.S["prop_topk"])

    def _run_stage2(self, sp):
        patches = self._patches()
        _show(pd.DataFrame({k: v["kind"].value_counts() for k, v in patches.items()}).T)
        PL.train_stage2(self.ctx, sp.name, sp.cfg, patches)

    def eval_patches(self, key, part):
        """(proposals, patch table) for key in {'mask', 'mask_anom', 'gt'} on val/test."""
        if key == "gt":
            return CS.gt_boxes(PL.part_df(self.ctx, part)), PL.build_eval_patches(self.ctx, "gt", None, part,
                                                                                  kind="gt")
        thr = self.proposal_threshold()
        kw = dict(bank_name="D1_clutter") if key == "mask_anom" else {}
        props = PL.proposals_from_preds(self.ctx, "S1_main", self._cfg("S1_main"), part, thr=thr,
                                        topk=self.S["prop_topk"], **kw)
        return props, PL.build_eval_patches(self.ctx, f"{key}_t{thr:.2f}", props, part)

    def prop_key(self, key):
        return "gt" if key == "gt" else f"{key}_t{self.proposal_threshold():.2f}"

    def _cascade(self, tag, s2_name, key, rules, **kw):
        _, pv = self.eval_patches(key, "val")
        _, pt = self.eval_patches(key, "test")
        return PL.evaluate_cascade(self.ctx, tag, "S1_main", self._cfg("S1_main"), s2_name, self._cfg(s2_name),
                                   self.prop_key(key), pv, pt, rules=rules, **kw)

    def _run_cascade1(self, sp):
        rows = []
        r, _, alpha = self._cascade("C1_mask", "S2_zoom", "mask", ("s1", "s2", "avg", "max"),
                                    label_prefix="C1 mask proposals: ")
        rows += r
        r, _, _ = self._cascade("C2_oracle", "S2_zoom", "gt", ("s2", "avg"), label_prefix="C2 oracle (true boxes): ")
        rows += r
        for k in (1, 3):
            r, _, _ = self._cascade(f"C3_top{k}", "S2_zoom", "mask", ("avg",), topk=k,
                                    label_prefix=f"C3 top-{k} proposals: ")
            rows += r
        for cf in (1.2, 2.0):
            r, _, _ = self._cascade(f"C3_ctx{cf}", "S2_zoom", "mask", ("avg",), context=cf,
                                    label_prefix=f"C3 crop context {cf}x: ")
            rows += r
        casc = PL.upsert_results(self.ctx, rows, "cascade")
        # proposal recall + box AP
        dt = PL.part_df(self.ctx, "test")
        gt = CS.gt_boxes(dt)
        props_t, pdf_t = self.eval_patches("mask", "test")
        pr = EV.proposal_recall(props_t, gt)
        PL.upsert_results(self.ctx, [dict(run="mask", **pr, per_scan=float(np.mean([len(v) for v in props_t.values()])))],
                          "proposals")
        p1 = PL.predict_stage1(self.ctx, "S1_main", self._cfg("S1_main"), "test")["probs"]
        crop = PL.predict_stage2(self.ctx, "S2_zoom", self._cfg("S2_zoom"), pdf_t, f"{self.prop_key('mask')}_test")
        _, pdf_g = self.eval_patches("gt", "test")
        crop_g = PL.predict_stage2(self.ctx, "S2_zoom", self._cfg("S2_zoom"), pdf_g, "gt_test")
        rows = [dict(run="stage1_only", method="Stage 1 only (mask blobs + image-level classes)",
                     **{k: v for k, v in EV.box_ap(CS.s1_box_predictions(props_t, p1, dt["uid"].to_numpy()),
                                                   gt).items() if k != "per_class_ap"}),
                dict(run="cascade", method="Look -> Zoom cascade",
                     **{k: v for k, v in EV.box_ap(CS.box_predictions(pdf_t, crop), gt).items()
                        if k != "per_class_ap"}),
                dict(run="zoom_on_gt", method="Zoom on true boxes (classification upper bound)",
                     **{k: v for k, v in EV.box_ap(CS.box_predictions(pdf_g, crop_g), gt).items()
                        if k != "per_class_ap"})]
        PL.upsert_results(self.ctx, rows, "box_ap")
        save_json(dict(alpha=alpha, thr=self.proposal_threshold()), self.run_dir(sp.name) / "eval_done.json")
        print(f"fusion weight alpha (Stage 1 share), tuned on validation: {alpha:.2f}")
        cols = ["label", "mAP", "mAP_lo", "mAP_hi", "macro_f1", "bag_auroc", "clear_at_rec95", "share_with_props",
                "crops_per_scan"]
        _show(casc[[c for c in cols if c in casc]])
        _show(pd.DataFrame(rows))

    def _run_cascade2(self, sp):
        r, _, _ = self._cascade("C2_mask_anom", "S2_zoom", "mask_anom", ("avg", "max"),
                                label_prefix="C2 mask+anomaly proposals: ")
        PL.upsert_results(self.ctx, r, "cascade")
        props_t, _ = self.eval_patches("mask_anom", "test")
        pr = EV.proposal_recall(props_t, CS.gt_boxes(PL.part_df(self.ctx, "test")))
        PL.upsert_results(self.ctx, [dict(run="mask_anom", **pr,
                                          per_scan=float(np.mean([len(v) for v in props_t.values()])))], "proposals")
        save_json(dict(done=True), self.run_dir(sp.name) / "eval_done.json")
        _show(pd.DataFrame(r)[["label", "mAP", "mAP_lo", "mAP_hi", "bag_auroc", "crops_per_scan"]])

    def _run_cascade4(self, sp):
        r, _, _ = self._cascade("C4_s2_nomat", "S2_zoom_nomat", "mask", ("avg",),
                                label_prefix="C4 Stage 2 without material: ")
        PL.upsert_results(self.ctx, r, "cascade")
        save_json(dict(done=True), self.run_dir(sp.name) / "eval_done.json")
        _show(pd.DataFrame(r)[["label", "mAP", "mAP_lo", "mAP_hi", "bag_auroc"]])

    def _run_openset(self, sp):
        held = SP.class_ids(self.S["heldout"])
        oracle = next(((n, self._cfg(n)) for n in ("A0_ref", "S1_main") if self.state(n) == "done"), (None, None))
        table, scores, pops = PL.openset_eval(self.ctx, "E1_open_s1", self._cfg("E1_open_s1"), "E1_open_bank",
                                              held, *oracle)
        table.to_csv(self.ctx.results / "openset.csv", index=False)
        sel = pops["U"] | pops["B"]
        VZ.plot_roc({k: (pops["U"][sel], v[sel]) for k, v in scores.items()},
                    title=f"Unseen threats ({', '.join(self.S['heldout'])}) vs clean bags",
                    path=self.ctx.fig("openset_roc.png"))
        plt.show()
        save_json(dict(done=True, oracle=oracle[0]), self.run_dir(sp.name) / "eval_done.json")
        _show(table)

    def _run_yolo(self, sp):
        ctx, d = self.ctx, self.run_dir(sp.name)
        ydata = ctx.work / "yolo_data"
        yaml_path = ydata / "stcray.yaml"
        if not yaml_path.exists():
            yaml_path = BL.export_yolo({p: PL.part_df(ctx, p) for p in ("train", "val", "test")}, ctx.cache, ydata,
                                       ctx.cache_size)
        dev = 0 if torch.cuda.is_available() else "cpu"
        weights = BL.train_yolo(yaml_path, d, model="yolov8n.yaml" if self.S["smoke"] else "yolov8s.pt",
                                epochs=self.S["yolo_epochs"], imgsz=ctx.cache_size,
                                batch=4 if self.S["smoke"] else 16, workers=self.S["num_workers"], device=dev)
        dv, dt = PL.part_df(ctx, "val"), PL.part_df(ctx, "test")
        f = d / "pred.pkl"
        if f.exists():
            Pv, Pt, boxes = pd.read_pickle(f)
        else:
            Pv, _ = BL.yolo_predict(weights, dv, ctx.cache, ctx.cache_size, device=dev)
            Pt, boxes = BL.yolo_predict(weights, dt, ctx.cache, ctx.cache_size, device=dev)
            pd.to_pickle((Pv, Pt, boxes), f)
        Yv, Yt = IX.label_matrix(dv), IX.label_matrix(dt)
        row, _ = EV.results_row(sp.name, Yt, Pt, EV.tune_thresholds(Yv, Pv), Yt.sum(1) > 0, dict(label=sp.label),
                                n_boot=ctx.n_boot)
        row["box_mAP50"] = EV.box_ap(boxes, CS.gt_boxes(dt))["box_mAP50"]
        PL.upsert_results(ctx, [row], "stage1")
        box = EV.box_ap(boxes, CS.gt_boxes(dt))
        PL.upsert_results(ctx, [dict(run="yolo", method="B5 YOLOv8s", box_mAP50=box["box_mAP50"],
                                     recall=box["recall"])], "box_ap")
        _show(pd.DataFrame([row])[["label", "mAP", "mAP_lo", "mAP_hi", "bag_auroc", "box_mAP50"]])

    def _run_clip(self, sp):
        ctx, d = self.ctx, self.run_dir(sp.name)
        d.mkdir(parents=True, exist_ok=True)
        dv, dt = PL.part_df(ctx, "val"), PL.part_df(ctx, "test")
        f = d / "pred.npz"
        if f.exists():
            z = np.load(f)
            Pv, Pt, bag_t = z["Pv"], z["Pt"], z["bag_t"]
        else:
            pre = None if self.S["smoke"] else "laion2b_s34b_b88k"
            Pv, _ = BL.clip_zero_shot(dv, ctx.cache, ctx.cache_size, pretrained=pre)
            Pt, bag_t = BL.clip_zero_shot(dt, ctx.cache, ctx.cache_size, pretrained=pre)
            np.savez(f, Pv=Pv, Pt=Pt, bag_t=bag_t)
        Yv, Yt = IX.label_matrix(dv), IX.label_matrix(dt)
        row, _ = EV.results_row(sp.name, Yt, Pt, EV.tune_thresholds(Yv, Pv), Yt.sum(1) > 0,
                                dict(label=sp.label,
                                     bag_auroc_benign_prompt=EV.bag_report(Yt.sum(1) > 0, bag_t).get("auroc")),
                                n_boot=ctx.n_boot)
        PL.upsert_results(ctx, [row], "stage1")
        _show(pd.DataFrame([row])[["label", "mAP", "mAP_lo", "mAP_hi", "bag_auroc", "bag_auroc_benign_prompt"]])

    # =============================================================================== analysis (any time)
    def headline(self):
        tables = [t for t in (self._table("stage1"), self._table("cascade")) if t is not None]
        a = self._table("ablations")
        if a is not None:
            tables.append(a[a.run == "A0_ref"])
        if not tables:
            print("no results yet - run some experiments first")
            return None
        master = pd.concat(tables, ignore_index=True).drop_duplicates("run", keep="last")
        master = master.sort_values("mAP", ascending=False)
        master.to_csv(self.ctx.results / "master_table.csv", index=False)
        cols = ["label", "mAP", "mAP_lo", "mAP_hi", "macro_auroc", "macro_f1", "micro_f1", "bag_auroc",
                "bag_auroc_lo", "bag_auroc_hi", "tpr_at_fpr5", "clear_at_rec95", "clear_at_rec99", "mask_dice",
                "pointing", "box_mAP50", "ms_per_scan", "params_m", "train_min"]
        order = ["B1_resnet50", "B2_convnext_nano", "B3_effnet_b4", "B4_vit_s16_384", "B5_yolov8s", "B6_clip",
                 "S1_main", "C1_mask_s2", "C1_mask_avg"]
        head = master[master.run.isin(order)].copy()
        head["o"] = head.run.map({r: i for i, r in enumerate(order)})
        head = head.sort_values("o").drop(columns="o")
        head.to_csv(self.ctx.results / "headline_table.csv", index=False)
        display(Markdown("**Headline comparison** (finished runs only)"))
        _show(head[[c for c in cols if c in head]])
        display(Markdown("**Every evaluated run**"))
        _show(master[[c for c in cols if c in master]])
        for name in ("proposals", "box_ap"):
            t = self._table(name)
            if t is not None:
                display(Markdown(f"**{name.replace('_', ' ')}**"))
                _show(t)
        return master

    def ablation_summary(self):
        d = self.update_deltas()
        a = self._table("ablations")
        if a is None:
            print("no ablation has finished yet")
            return
        cols = ["label", "mAP", "mAP_lo", "mAP_hi", "val_mAP", "macro_f1", "bag_auroc", "clear_at_rec95",
                "mask_dice", "pointing", "params_m", "train_min"]
        _show(a[[c for c in cols if c in a]])
        if d is not None and len(d):
            _show(d)
            VZ.plot_deltas(d, "A0", path=self.ctx.fig("ablation_deltas.png"))
            plt.show()
        missing = [n for n in self.names("ablation") if self.state(n) != "done"]
        if missing:
            print("still to run:", ", ".join(missing))

    def anomaly_summary(self):
        t = self._table("anomaly")
        if t is None:
            print("no anomaly bank has been evaluated yet")
            return
        _show(t)

    def fused(self):
        """Cascade scores (val, test) if C1_fusion is done, else None."""
        f = self.run_dir("C1_fusion") / "eval_done.json"
        if not f.exists():
            return None
        alpha = load_json(f)["alpha"]
        key = self.prop_key("mask")
        out = {}
        for part in ("val", "test"):
            _, pdf = self.eval_patches("mask", part)
            p1, p2, has, _ = PL.cascade_scores(self.ctx, "S1_main", self._cfg("S1_main"), "S2_zoom",
                                               self._cfg("S2_zoom"), part, key, pdf)
            out[part] = CS.fuse(p1, p2, has, alpha, "avg")
        return out

    def model_preds(self):
        """{name: (val_probs, test_probs)} for the models used in the analysis plots."""
        preds = {}
        if self.state("S1_main") == "done":
            preds["CAMO-X Stage 1"] = tuple(PL.predict_stage1(self.ctx, "S1_main", self._cfg("S1_main"), p)["probs"]
                                            for p in ("val", "test"))
        if self.state("B2_convnext_nano") == "done":
            preds["B2 ConvNeXt-Nano"] = tuple(PL.predict_stage1(self.ctx, "B2_convnext_nano",
                                                                self._cfg("B2_convnext_nano"), p)["probs"]
                                              for p in ("val", "test"))
        fu = self.fused()
        if fu is not None:
            preds["CAMO-X cascade"] = (fu["val"], fu["test"])
        return preds

    def per_class_and_curves(self):
        preds = self.model_preds()
        if not preds:
            print("needs S1_main (and optionally B2_convnext_nano, C1_fusion)")
            return
        dv, dt = PL.part_df(self.ctx, "val"), PL.part_df(self.ctx, "test")
        Yv, Yt = IX.label_matrix(dv), IX.label_matrix(dt)
        ap = {k: EV.per_class_ap(Yt, v[1]) for k, v in preds.items()}
        VZ.plot_per_class({k: np.nan_to_num(v) for k, v in ap.items()}, path=self.ctx.fig("per_class_ap.png"))
        plt.show()
        _show(pd.DataFrame(ap, index=CLASS_NAMES).T, 3)
        VZ.plot_recall_clearance(Yt.sum(1) > 0, {k: v[1].max(1) for k, v in preds.items()},
                                 path=self.ctx.fig("recall_vs_clearance.png"))
        plt.show()
        key = list(preds)[-1]
        cm = EV.confusion_single(Yt, preds[key][1])
        VZ.plot_confusion(cm, path=self.ctx.fig("confusion.png"),
                          title=f"Confusion on single-threat test scans ({key})")
        plt.show()
        rows = []
        for g, names in LOOKALIKE_GROUPS.items():
            ids = [CLASS_NAMES.index(n) for n in names]
            sub = cm[np.ix_(ids, range(NUM_CLASSES))]
            within = cm[np.ix_(ids, ids)]
            rows.append(dict(group=g, scans=int(sub.sum()), correct=float(np.trace(within) / max(sub.sum(), 1)),
                             confused_within_group=float((within.sum() - np.trace(within)) / max(sub.sum(), 1))))
        _show(pd.DataFrame(rows), 3)

    def strata(self):
        preds = self.model_preds()
        if not preds:
            print("needs S1_main")
            return
        dv, dt = PL.part_df(self.ctx, "val"), PL.part_df(self.ctx, "test")
        Yv, Yt = IX.label_matrix(dv), IX.label_matrix(dt)
        thr = {k: EV.tune_thresholds(Yv, v[0]) for k, v in preds.items()}
        tables, edges = PL.strata_tables(self.ctx, {k: v[1] for k, v in preds.items()}, thr)
        print("tercile edges:", {k: np.round(v, 4).tolist() for k, v in edges.items()})
        for k, t in tables.items():
            print(k)
            _show(pd.concat(t, names=["axis"]), 3)
        VZ.plot_strata(tables, "recall", path=self.ctx.fig("strata_recall.png"))
        plt.show()
        levels = PL.caption_strata(dt)
        if levels is not None:
            rows = []
            for name, (_, P) in preds.items():
                for lvl, lname in enumerate(["limited", "medium", "heavy", "extreme"]):
                    m = levels == lvl
                    if m.sum() >= 10:
                        rec = float(((P[m] >= thr[name]) & (Yt[m] > 0)).sum() / max(Yt[m].sum(), 1))
                        rows.append(dict(model=name, clutter_from_caption=lname, scans=int(m.sum()),
                                         mAP=EV.mean_ap(Yt[m], P[m]), recall=rec))
            _show(pd.DataFrame(rows), 3)
        else:
            print("captions do not state a clutter level often enough - caption strata skipped")
        nd = dt["near_dup_of_train"].to_numpy().astype(bool)
        if nd.any():
            _show(pd.DataFrame([dict(model=k, all_test=EV.mean_ap(Yt, v[1]),
                                     without_near_duplicates=EV.mean_ap(Yt[~nd], v[1][~nd]))
                                for k, v in preds.items()]))

    def latency(self):
        if self.state("S1_main") != "done":
            print("needs S1_main")
            return
        s2 = "S2_zoom" if self.state("S2_zoom") == "done" else None
        bank = "D1_clutter" if self.state("D1_clutter") == "done" else None
        lat = PL.latency_table(self.ctx, "S1_main", self._cfg("S1_main"), s2, self._cfg(s2) if s2 else None, bank,
                               n=10 if self.S["smoke"] else 50)
        casc = self._table("cascade")
        if s2 and casc is not None and (casc.run == "C1_mask_avg").any():
            cps = float(casc[casc.run == "C1_mask_avg"]["crops_per_scan"].iloc[0])
            per_crop = float(lat.loc[lat.stage.str.startswith("Stage 2"), "ms"].iloc[0])
            lat.loc[len(lat)] = dict(stage=f"Stage 2 total at {cps:.2f} crops per scan", ms=per_crop * cps)
        lat.to_csv(self.ctx.results / "latency.csv", index=False)
        _show(lat, 2)

    def seed_summary(self):
        s1, seeds = self._table("stage1"), self._table("seeds")
        if seeds is None or s1 is None:
            print("no seed runs yet (S1_main_seed1/2, B2_convnext_nano_seed1/2)")
            return
        allr = pd.concat([s1[s1.run.isin(["S1_main", "B2_convnext_nano"])], seeds])
        allr["model"] = np.where(allr.run.str.startswith("B2"), "B2 ConvNeXt-Nano", "CAMO-X Stage 1")
        _show(allr.groupby("model")["mAP"].agg(["mean", "std", "count"]))

    def qualitative(self, n_each=3):
        if self.state("S1_main") != "done":
            print("needs S1_main")
            return
        ctx = self.ctx
        preds = self.model_preds()
        key = "CAMO-X cascade" if "CAMO-X cascade" in preds else "CAMO-X Stage 1"
        P = preds[key][1]
        dt = PL.part_df(ctx, "test")
        Yt = IX.label_matrix(dt)
        is_thr = Yt.sum(1) > 0
        cfg_main = self._cfg("S1_main")
        pt = PL.predict_stage1(ctx, "S1_main", cfg_main, "test")
        model = EN.load_run_model(cfg_main, self.run_dir("S1_main"), get_device())
        anom = None
        if self.state("D1_clutter") == "done":
            anom = np.load(self.run_dir("D1_clutter") / "scores_test.npz")["maps"]
        props = pdf = crop_probs = None
        if self.state("C1_fusion") == "done":
            props, pdf = self.eval_patches("mask", "test")
            crop_probs = PL.predict_stage2(ctx, "S2_zoom", self._cfg("S2_zoom"), pdf,
                                           f"{self.prop_key('mask')}_test")
        s2_ctx = self._cfg("S2_zoom")["crop_context"]
        device = get_device()

        def show(i, title):
            from .utils import boxes_orig_to_cache
            r = dt.iloc[i]
            img, m = CA.load_cached(r.uid, ctx.cache, ctx.cache_size)
            x = torch.from_numpy(img).permute(2, 0, 1)[None]
            if cfg_main["img_size"] != ctx.cache_size:
                x = torch.nn.functional.interpolate(x.float(), size=(cfg_main["img_size"],) * 2,
                                                    mode="area").round().byte()
            with torch.no_grad():
                out = model(x.to(device))
            top = np.argsort(-P[i])[:2]
            cams = [out["cams"][0, c].float().cpu().numpy() for c in top]
            names = [f"{CLASS_NAMES[c]} {P[i, c]:.2f}" for c in top]
            props_c, crops, titles = [], [], []
            if props is not None:
                for b in props.get(r.uid, []):
                    props_c.append(boxes_orig_to_cache([b[:4] + [0]], r.scale, r.pad_x, r.pad_y)[0][:4].tolist())
                for j in np.flatnonzero(pdf["uid"].to_numpy() == r.uid)[:3]:
                    patch = imread_rgb(pdf.iloc[j]["path"])
                    side = s2_ctx * CA.PATCH_SIZE / CA.PATCH_CONTEXT
                    c0 = CA.PATCH_SIZE / 2
                    crops.append(crop_with_pad(patch, (c0 - side / 2, c0 - side / 2, c0 + side / 2, c0 + side / 2),
                                               160))
                    k = int(np.argmax(crop_probs[j]))
                    titles.append(f"zoom: {CLASS_NAMES[k]} {crop_probs[j, k]:.2f}")
            truth = ", ".join(CLASS_NAMES[c] for c in r.labels) or "clean bag"
            seg = pt["seg"][i] if pt.get("seg") is not None else np.zeros((8, 8), np.uint8)
            VZ.qualitative_panel(img, m, seg, cams, names, None if anom is None else anom[i].astype(np.float32),
                                 props_c, crops, titles, title=f"{title} | truth: {truth} ({key})",
                                 path=ctx.fig(f"qual_{title.replace(' ', '_')}_{r.uid}.png"))
            plt.show()

        worst_true = np.nanmin(np.where(Yt > 0, P, np.inf), axis=1)
        hits = [int(np.nanargmax(np.where(Yt[:, c] > 0, P[:, c], -1))) for c in np.argsort(-Yt.sum(0))[:n_each]
                if Yt[:, c].sum()]
        misses = [int(i) for i in np.argsort(np.where(is_thr, worst_true, np.inf))[:n_each] if is_thr[i]]
        alarms = [int(i) for i in np.argsort(-np.where(~is_thr, P.max(1), -1))[:n_each] if not is_thr[i]]
        for i in hits:
            show(i, "confident hit")
        for i in misses:
            show(i, "worst miss")
        for i in alarms:
            show(i, "worst false alarm")
        imgs, titles, boxes = [], [], []
        for i in [int(i) for i in np.argsort(np.where(is_thr, worst_true, np.inf))[:6] if is_thr[i]]:
            r = dt.iloc[i]
            imgs.append(CA.load_cached(r.uid, ctx.cache, ctx.cache_size)[0])
            titles.append(f"{', '.join(CLASS_NAMES[c] for c in r.labels)} | score {worst_true[i]:.2f}")
            boxes.append(r.boxes_cache)
        VZ.gallery(imgs, titles, ncols=6, title="Missed threats (green = true boxes)", boxes=boxes,
                   path=ctx.fig("gallery_misses.png"))
        plt.show()
        if self.state("E1_eval") == "done":
            held = SP.class_ids(self.S["heldout"])
            _, scores, pops = PL.openset_eval(ctx, "E1_open_s1", self._cfg("E1_open_s1"), "E1_open_bank", held)
            s_f = scores["fused (max z)"]
            U = np.flatnonzero(pops["U"])
            order = U[np.argsort(-s_f[U])]
            thr5 = np.quantile(s_f[pops["B"]], 0.95) if pops["B"].sum() else np.inf
            imgs, titles, boxes = [], [], []
            for i in list(order[:4]) + list(order[-4:]):
                r = dt.iloc[i]
                imgs.append(CA.load_cached(r.uid, ctx.cache, ctx.cache_size)[0])
                titles.append(("CAUGHT " if s_f[i] >= thr5 else "missed ") + f"z={s_f[i]:.1f} " +
                              ", ".join(CLASS_NAMES[c] for c in r.labels))
                boxes.append(r.boxes_cache)
            VZ.gallery(imgs, titles, ncols=4, title="Unseen threats: highest and lowest fused scores "
                       "(alarm threshold = 5% false alarms on clean bags)", boxes=boxes,
                       path=ctx.fig("gallery_unseen.png"))
            plt.show()

    def export(self):
        for f in sorted(self.ctx.results.glob("*.csv")):
            t = pd.read_csv(f)
            num = t.select_dtypes("number").columns
            t[num] = t[num].round(4)
            try:
                (self.ctx.results / f"{f.stem}.md").write_text(t.to_markdown(index=False), encoding="utf-8")
            except ImportError:
                (self.ctx.results / f"{f.stem}.md").write_text(t.to_string(index=False), encoding="utf-8")
            try:
                (self.ctx.results / f"{f.stem}.tex").write_text(t.to_latex(index=False, float_format="%.4f"),
                                                                encoding="utf-8")
            except Exception:
                pass
        print("tables:", [p.name for p in sorted(self.ctx.results.glob("*.csv"))])
        print("figures:", [p.name for p in sorted(self.ctx.figs.glob("*.png"))])


__all__ = ["prepare_data", "annotation_audit", "eda", "split_summary", "material_check", "tip_preview", "Study",
           "Spec"]
