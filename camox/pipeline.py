"""High-level experiment helpers used by the notebook (keeps the notebook cells short)."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from . import anomaly as AN
from . import cascade as CS
from . import evaluate as EV
from .cache import PATCH_CONTEXT, PATCH_SIZE, build_patches, box_size_pool, cache_paths, load_cached
from .config import CLASS_NAMES, NUM_CLASSES
from .datasets import CropDataset, Stage1Dataset
from .engine import load_run_model, make_loader, measure_latency, predict, train_model
from .indexing import label_matrix
from .splits import class_ids, train_subset
from .utils import cuda_sync, get_device, imread_rgb, load_json, save_json


@dataclass
class Ctx:
    work: Path
    cache_size: int = 512
    smoke: bool = False
    num_workers: int = 4
    n_boot: int = 1000
    df: pd.DataFrame | None = None
    donors: pd.DataFrame | None = None
    notes: dict = field(default_factory=dict)

    def __post_init__(self):
        self.work = Path(self.work)
        for d in (self.cache, self.runs, self.figs, self.results):
            d.mkdir(parents=True, exist_ok=True)

    @property
    def cache(self):
        return self.work / "cache"

    @property
    def runs(self):
        return self.work / "runs"

    @property
    def figs(self):
        return self.work / "figures"

    @property
    def results(self):
        return self.work / "results"

    def fig(self, name):
        return self.figs / name


# ----------------------------------------------------------------------------- data access
def part_df(ctx, part, exclude_ids=(), only_ok=True):
    d = ctx.df[ctx.df["part"] == part]
    if only_ok and "cache_ok" in d:
        d = d[d["cache_ok"] == True]  # noqa: E712
    if exclude_ids:
        ex = set(int(c) for c in exclude_ids)
        d = d[[not (ex & set(labs)) for labs in d["labels"]]]
    return d.reset_index(drop=True)


def items_of(d):
    return [(r.uid, [int(c) for c in r.labels], [list(b) for b in (r.boxes_cache or [])])
            for r in d.itertuples(index=False)]


def excluded_ids(cfg):
    return class_ids(cfg.get("exclude_classes", ()) or ())


def gt_masks(ctx, part, res):
    """(N,res,res) uint8 ground-truth masks for a part (cached as .npy)."""
    f = ctx.cache / f"gtmask_{part}_{res}.npy"
    d = part_df(ctx, part)
    if f.exists():
        arr = np.load(f)
        if len(arr) == len(d):
            return arr
    arr = np.zeros((len(d), res, res), np.uint8)
    for i, u in enumerate(d["uid"]):
        _, m = load_cached(u, ctx.cache, ctx.cache_size)
        if m is not None:
            arr[i] = cv2.resize(m, (res, res), interpolation=cv2.INTER_AREA)
    np.save(f, arr)
    return arr


def _bs_eval(cfg):
    return 8 if cfg["img_size"] <= 128 else (32 if cfg["img_size"] <= 384 else 24)


# ----------------------------------------------------------------------------- Stage 1
def stage1_train_df(ctx, cfg):
    tr = part_df(ctx, "train", excluded_ids(cfg))
    frac = float(cfg.get("train_fraction", 1.0))
    if frac < 1.0:
        keep = train_subset(len(tr), frac, seed=0, Y=label_matrix(tr))  # same subset for every ablation
        tr = tr.iloc[keep].reset_index(drop=True)
    return tr


def train_stage1(ctx, name, cfg, log=print):
    ex = excluded_ids(cfg)
    tr = stage1_train_df(ctx, cfg)
    va = part_df(ctx, "val", ex)
    ds_tr = Stage1Dataset(items_of(tr), ctx.cache, ctx.cache_size, cfg["img_size"], train=True,
                          tip=cfg.get("tip", "none"), tip_prob=cfg.get("tip_prob", 0.5), donors=ctx.donors,
                          hue_jitter=cfg.get("hue_jitter", False), geo_aug=cfg.get("geo_aug", True),
                          input_mode=cfg.get("input_mode", "rgb"), excluded=ex)
    ds_va = Stage1Dataset(items_of(va), ctx.cache, ctx.cache_size, cfg["img_size"], train=False,
                          input_mode=cfg.get("input_mode", "rgb"))
    return train_model(cfg, ds_tr, ds_va, ctx.runs / name, pos_counts=label_matrix(tr).sum(0), log=log)


def predict_stage1(ctx, name, cfg, part, want_seg=True, cam_peaks=True, force=False):
    f = ctx.runs / name / f"pred_{part}.npz"
    d = part_df(ctx, part)
    if f.exists() and not force:
        z = np.load(f, allow_pickle=True)
        out = {k: z[k] for k in z.files}
        if len(out["probs"]) == len(d):
            for k in ("seg", "cam_peaks"):
                if k in out and out[k].dtype == object:
                    out[k] = None
            g = int(out.get("cam_grid", -1))
            out["cam_grid"] = g if g > 0 else None
            return out
    device = get_device()
    model = load_run_model(cfg, ctx.runs / name, device)
    ds = Stage1Dataset(items_of(d), ctx.cache, ctx.cache_size, cfg["img_size"], train=False,
                       input_mode=cfg.get("input_mode", "rgb"))
    loader = make_loader(ds, _bs_eval(cfg), False, ctx.num_workers)
    res = predict(model, loader, device, cfg.get("amp", True), cfg.get("tta_flip", False), want_seg,
                  cam_peaks and cfg.get("arch", "camox") == "camox", progress=True)
    res["uids"] = d["uid"].to_numpy()
    np.savez_compressed(f, probs=res["probs"], targets=res["targets"], uids=res["uids"],
                        seg=res["seg"] if res["seg"] is not None else np.array(None),
                        cam_peaks=res["cam_peaks"] if res["cam_peaks"] is not None else np.array(None),
                        cam_grid=res["cam_grid"] if res["cam_grid"] is not None else -1)
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return res


def _known(cfg):
    ex = set(excluded_ids(cfg))
    return [c for c in range(NUM_CLASSES) if c not in ex]


def evaluate_stage1(ctx, name, cfg, label=None, latency=True):
    """Test-set row for one Stage-1 (or baseline) run. Thresholds are tuned on validation."""
    known = _known(cfg)
    ex = excluded_ids(cfg)
    pv = predict_stage1(ctx, name, cfg, "val")
    pt = predict_stage1(ctx, name, cfg, "test")
    dv, dt = part_df(ctx, "val"), part_df(ctx, "test")
    kv = np.array([not (set(ex) & set(l)) for l in dv["labels"]])
    kt = np.array([not (set(ex) & set(l)) for l in dt["labels"]])
    thr = np.full(NUM_CLASSES, 0.5, np.float32)
    thr[known] = EV.tune_thresholds(pv["targets"][kv][:, known], pv["probs"][kv][:, known])
    Yt, Pt = pt["targets"][kt], pt["probs"][kt]
    is_threat = Yt[:, known].sum(1) > 0
    done = load_json(ctx.runs / name / "done.json") if (ctx.runs / name / "done.json").exists() else {}
    extra = dict(label=label or name, val_mAP=EV.mean_ap(pv["targets"][kv][:, known], pv["probs"][kv][:, known]),
                 params_m=done.get("params_m"), train_min=done.get("train_minutes"),
                 peak_vram_gb=done.get("peak_vram_gb"), epochs=done.get("epochs_run"))
    row, rep = EV.results_row(name, Yt, Pt, thr, is_threat, extra, n_boot=ctx.n_boot, classes=known)
    if pt.get("seg") is not None:
        res = pt["seg"].shape[-1]
        gm = gt_masks(ctx, "test", res)[kt]
        sr = EV.seg_report(pt["seg"][kt], gm, is_threat)
        row.update(mask_dice=sr["dice"], mask_iou=sr["iou"])
    if pt.get("cam_peaks") is not None and pt.get("cam_grid"):
        pg = EV.pointing_game(pt["cam_peaks"][kt], pt["cam_grid"], dt["boxes_cache"][kt].tolist(),
                              dt["labels"][kt].tolist(), ctx.cache_size)
        row.update(pointing=pg["pointing_acc"])
    if latency:
        try:
            device = get_device()
            model = load_run_model(cfg, ctx.runs / name, device)
            row["ms_per_scan"] = measure_latency(model, cfg["img_size"], device, n=10 if ctx.smoke else 50)
            del model
        except Exception as e:  # noqa: BLE001
            print("latency failed:", e)
    save_json(dict(per_class_ap=rep["per_class_ap"], thresholds=thr, known=known),
              ctx.runs / name / "test_metrics.json")
    return row


def upsert_results(ctx, rows, table="results"):
    f = ctx.results / f"{table}.csv"
    new = pd.DataFrame(rows)
    if f.exists():
        old = pd.read_csv(f)
        old = old[~old["run"].isin(new["run"])]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(f, index=False)
    return new


def paired_deltas(ctx, ref_name, ref_cfg, runs):
    """Paired bootstrap of test-mAP differences vs a reference run (same test scans)."""
    pr = predict_stage1(ctx, ref_name, ref_cfg, "test")
    rows = []
    for name, cfg, label in runs:
        if not (ctx.runs / name / "best.pt").exists():
            continue
        pb = predict_stage1(ctx, name, cfg, "test")
        d = EV.paired_delta(pr["targets"], pr["probs"], pb["probs"], n_boot=ctx.n_boot)
        rows.append(dict(run=name, label=label, delta=d["delta"], d_lo=d["lo"], d_hi=d["hi"],
                         p_not_better=d["p_le0"]))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- proposals (Look)
def _anomaly_z(ctx, bank_name, part):
    f = ctx.runs / bank_name / f"scores_{part}.npz"
    fit = ctx.runs / bank_name / "null_fit.json"
    if not f.exists() or not fit.exists():
        return None
    z = np.load(f)
    cell = load_json(fit)["cell_fit"]
    return (z["maps"].astype(np.float32) - cell[0]) / cell[1]


@torch.no_grad()
def proposals_stream(ctx, name, cfg, part, thr=0.35, topk=5):
    """Run Stage 1 over a (large) part and keep only proposals (no maps kept in RAM)."""
    d = part_df(ctx, part, excluded_ids(cfg))
    device = get_device()
    model = load_run_model(cfg, ctx.runs / name, device)
    ds = Stage1Dataset(items_of(d), ctx.cache, ctx.cache_size, cfg["img_size"], train=False,
                       input_mode=cfg.get("input_mode", "rgb"))
    loader = make_loader(ds, _bs_eval(cfg), False, ctx.num_workers)
    props = {}
    rows = list(d.itertuples(index=False))
    for x, _, _, idx in loader:
        x = x.to(device, non_blocking=True)
        if device.type == "cuda":
            x = x.contiguous(memory_format=torch.channels_last)
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            seg = model(x)["seg"]
        seg = (torch.sigmoid(seg.float())[:, 0] * 255).round().to(torch.uint8).cpu().numpy()
        for s, i in zip(seg, idx.numpy()):
            r = rows[int(i)]
            meta = dict(scale=r.scale, pad_x=r.pad_x, pad_y=r.pad_y, img_w=r.img_w, img_h=r.img_h)
            props[r.uid] = CS.extract_proposals(s, meta, thr=thr, topk=topk, cache_size=ctx.cache_size)
    return props


def proposals_from_preds(ctx, name, cfg, part, thr=0.35, topk=5, bank_name=None, z0=3.0, use_seg=True):
    pr = predict_stage1(ctx, name, cfg, part)
    d = part_df(ctx, part)
    z = _anomaly_z(ctx, bank_name, part) if bank_name else None
    seg = pr["seg"] if use_seg else None
    return CS.proposals_for(d, seg, z, thr=thr, topk=topk, z0=z0, cache_size=ctx.cache_size)


def sweep_proposal_thr(ctx, name, cfg, part="val", grid=(0.2, 0.3, 0.4, 0.5, 0.6), topk=5, max_per_scan=3.0,
                       bank_name=None):
    """Pick the mask threshold with the best GT coverage while keeping proposals per scan <= max_per_scan."""
    d = part_df(ctx, part)
    gts = CS.gt_boxes(d)
    rows = []
    for t in grid:
        props = proposals_from_preds(ctx, name, cfg, part, thr=t, topk=topk, bank_name=bank_name)
        pr = EV.proposal_recall(props, gts)
        rows.append(dict(thr=t, recall_cover=pr["prop_recall_cover"], recall_iou50=pr["prop_recall_iou50"],
                         per_scan=float(np.mean([len(v) for v in props.values()])) if props else 0.0))
    tab = pd.DataFrame(rows)
    ok = tab[tab["per_scan"] <= max_per_scan]
    pick = (ok if len(ok) else tab).sort_values(["recall_cover", "thr"], ascending=[False, False]).iloc[0]
    return float(pick["thr"]), tab


# ----------------------------------------------------------------------------- Stage 2 (Zoom)
def build_stage2_patches(ctx, s1_name, s1_cfg, tag="main", thr=0.35, topk=5, n_random=1):
    """Training/validation patches: GT boxes + random background + Stage-1 false alarms (hard negatives)."""
    out = {}
    ex = excluded_ids(s1_cfg)
    sizes = box_size_pool(part_df(ctx, "train", ex))
    for part in ("train", "val"):
        f = ctx.cache / f"patches_{tag}_{part}.pkl"
        if f.exists():
            out[part] = pd.read_pickle(f)
            continue
        props = proposals_stream(ctx, s1_name, s1_cfg, part, thr=thr, topk=topk)
        d = part_df(ctx, part, ex)
        recs = d[["uid", "img_path", "boxes"]].to_dict("records")
        pdf = build_patches(recs, ctx.cache / f"patches_{tag}" / part, "train", proposals=props,
                            n_random=n_random, sizes=sizes, seed=0 if part == "train" else 1,
                            workers=max(2, ctx.num_workers * 2))
        pdf.to_pickle(f)
        out[part] = pdf
    return out


def build_eval_patches(ctx, key, props, part, kind="eval"):
    f = ctx.cache / f"evalpatches_{key}_{part}.pkl"
    if f.exists():
        return pd.read_pickle(f)
    d = part_df(ctx, part)
    recs = d[["uid", "img_path", "boxes"]].to_dict("records")
    pdf = build_patches(recs, ctx.cache / f"evalpatches_{key}" / part, kind, proposals=props,
                        workers=max(2, ctx.num_workers * 2))
    if len(pdf) == 0:
        pdf = pd.DataFrame(columns=["path", "uid", "kind", "k", "boxes", "score", "region", "box"])
    pdf.to_pickle(f)
    return pdf


def train_stage2(ctx, name, cfg, patches, log=print):
    tr = patches["train"]
    va = patches["val"]
    rows_tr = list(zip(tr["path"], tr["boxes"]))
    rows_va = list(zip(va["path"], va["boxes"]))
    ex = excluded_ids(cfg)
    ds_tr = CropDataset(rows_tr, cfg["img_size"], True, cfg["crop_context"], cfg["train_context"],
                        cfg["train_jitter"], cfg.get("input_mode", "rgb"), cfg.get("hue_jitter", False), ex)
    ds_va = CropDataset(rows_va, cfg["img_size"], False, cfg["crop_context"], input_mode=cfg.get("input_mode", "rgb"),
                        excluded=ex)
    pos = np.zeros(NUM_CLASSES)
    for bl in tr["boxes"]:
        for c in {int(b[4]) for b in bl if b[4] >= 0}:
            pos[c] += 1
    return train_model(cfg, ds_tr, ds_va, ctx.runs / name, pos_counts=pos, log=log)


def predict_stage2(ctx, name, cfg, pdf, key, context=None):
    context = context or cfg["crop_context"]
    f = ctx.runs / name / f"crops_{key}_{context:.2f}.npy"
    if f.exists():
        arr = np.load(f)
        if len(arr) == len(pdf):
            return arr
    if len(pdf) == 0:
        return np.zeros((0, NUM_CLASSES), np.float32)
    device = get_device()
    model = load_run_model(cfg, ctx.runs / name, device)
    ds = CropDataset(list(zip(pdf["path"], pdf["boxes"])), cfg["img_size"], False, context,
                     input_mode=cfg.get("input_mode", "rgb"))
    res = predict(model, make_loader(ds, 64 if not ctx.smoke else 8, False, ctx.num_workers), device,
                  cfg.get("amp", True), want_seg=False, progress=True)
    np.save(f, res["probs"])
    del model
    return res["probs"]


def cascade_scores(ctx, s1_name, s1_cfg, s2_name, s2_cfg, part, props_key, pdf, context=None, topk=None):
    p1 = predict_stage1(ctx, s1_name, s1_cfg, part)["probs"]
    crop = predict_stage2(ctx, s2_name, s2_cfg, pdf, f"{props_key}_{part}", context)
    uids = part_df(ctx, part)["uid"].to_numpy()
    p2, has = CS.aggregate_crops(crop, pdf["uid"].to_numpy() if len(pdf) else [], uids, topk,
                                 pdf["k"].to_numpy() if len(pdf) else None)
    return p1, p2, has, crop


def evaluate_cascade(ctx, tag, s1_name, s1_cfg, s2_name, s2_cfg, props_key, pdf_val, pdf_test, rules=("s1", "s2", "avg", "max"),
                     context=None, topk=None, label_prefix=""):
    known = _known(s1_cfg)
    p1v, p2v, hv, _ = cascade_scores(ctx, s1_name, s1_cfg, s2_name, s2_cfg, "val", props_key, pdf_val, context, topk)
    p1t, p2t, ht, crop_t = cascade_scores(ctx, s1_name, s1_cfg, s2_name, s2_cfg, "test", props_key, pdf_test, context,
                                          topk)
    Yv = predict_stage1(ctx, s1_name, s1_cfg, "val")["targets"]
    Yt = predict_stage1(ctx, s1_name, s1_cfg, "test")["targets"]
    alpha, _ = CS.tune_alpha(Yv, p1v, p2v, hv, classes=known)
    oracle = props_key.startswith("gt")
    rows, fused = [], {}
    for rule in rules:
        fv = CS.fuse(p1v, p2v, hv, alpha, rule)
        ft = CS.fuse(p1t, p2t, ht, alpha, rule)
        thr = np.full(NUM_CLASSES, 0.5, np.float32)
        thr[known] = EV.tune_thresholds(Yv[:, known], fv[:, known])
        row, rep = EV.results_row(f"{tag}_{rule}", Yt, ft, thr, Yt[:, known].sum(1) > 0,
                                  dict(label=f"{label_prefix}{rule}", alpha=alpha if rule == "avg" else None,
                                       share_with_props=float(ht.mean()),
                                       crops_per_scan=float(len(crop_t) / max(1, len(Yt)))),
                                  n_boot=ctx.n_boot, classes=known)
        if oracle:  # proposals come from the labels, so bag-level screening numbers are meaningless
            for k in list(row):
                if k.startswith(("bag_", "tpr_at_", "clear_at_")):
                    row[k] = np.nan
        rows.append(row)
        fused[rule] = ft
    return rows, fused, alpha


# ----------------------------------------------------------------------------- anomaly branch
def build_bank(ctx, name, mode="clutter", backbone="wide_resnet50_2.tv_in1k", layers="l23", bank_size=16000,
               coreset=True, per_image=16, exclude_ids=(), img_size=None, seed=0):
    f = ctx.runs / name / "bank.pt"
    if f.exists():
        return torch.load(f, weights_only=True)
    (ctx.runs / name).mkdir(parents=True, exist_ok=True)
    img_size = img_size or ctx.cache_size
    device = get_device()
    ext = AN.PatchFeatures(backbone, layers, pretrained=not ctx.smoke).to(device)
    tr = part_df(ctx, "train", exclude_ids)
    if mode == "clean":  # only clean bags contribute, so do not read the other scans at all
        tr = tr[[len(l) == 0 for l in tr["labels"]]].reset_index(drop=True)
    ds = Stage1Dataset(items_of(tr), ctx.cache, ctx.cache_size, img_size, train=False)
    loader = make_loader(ds, 16 if not ctx.smoke else 8, False, ctx.num_workers)
    t = time.time()
    feats = AN.collect_bank(ext, loader, device, mode=mode, per_image=per_image, seed=seed)
    bank = AN.greedy_coreset(feats, bank_size, device, seed=seed) if coreset else \
        AN.random_subsample(feats, bank_size, seed)
    torch.save(bank, f)
    save_json(dict(mode=mode, backbone=backbone, layers=layers, bank_size=int(len(bank)), coreset=coreset,
                   collected=int(len(feats)), minutes=(time.time() - t) / 60, img_size=img_size,
                   exclude=list(exclude_ids)), ctx.runs / name / "bank.json")
    return bank


def score_bank(ctx, name, part, backbone="wide_resnet50_2.tv_in1k", layers="l23", img_size=None):
    f = ctx.runs / name / f"scores_{part}.npz"
    if f.exists():
        z = np.load(f)
        return {k: z[k] for k in z.files}
    bank = build_bank(ctx, name)
    img_size = img_size or ctx.cache_size
    device = get_device()
    ext = AN.PatchFeatures(backbone, layers, pretrained=not ctx.smoke).to(device)
    d = part_df(ctx, part)
    ds = Stage1Dataset(items_of(d), ctx.cache, ctx.cache_size, img_size, train=False)
    res = AN.score_loader(ext, bank, make_loader(ds, 16 if not ctx.smoke else 8, False, ctx.num_workers), device)
    np.savez_compressed(f, **res)
    return res


def evaluate_bank(ctx, name, label, backbone="wide_resnet50_2.tv_in1k", layers="l23", exclude_ids=()):
    sv = score_bank(ctx, name, "val", backbone, layers)
    st = score_bank(ctx, name, "test", backbone, layers)
    dv, dt = part_df(ctx, "val"), part_df(ctx, "test")
    gv = gt_masks(ctx, "val", ctx.cache_size // 4)
    gt = gt_masks(ctx, "test", ctx.cache_size // 4)
    thr_v = np.array([len(l) > 0 for l in dv["labels"]])
    # null distributions from validation: clean bags + threat-free cells of threat bags
    null = np.concatenate([sv["score_max"][~thr_v], AN.masked_max(sv["maps"][thr_v], gv[thr_v])])
    img_fit = AN.robust_fit(null)
    cell_fit = AN.clutter_cell_stats(sv["maps"], gv)
    save_json(dict(img_fit=img_fit, cell_fit=cell_fit, n_null=int(np.isfinite(null).sum())),
              ctx.runs / name / "null_fit.json")
    ex = set(int(c) for c in exclude_ids)
    keep = np.array([not (ex & set(l)) for l in dt["labels"]])
    is_threat = np.array([len(l) > 0 for l in dt["labels"]])[keep]
    row = dict(run=name, label=label)
    for key in ("score_max", "score_topk"):
        b = EV.bag_report(is_threat, st[key][keep])
        row[f"bag_auroc_{key[6:]}"] = b.get("auroc")
        row[f"tpr_at_fpr5_{key[6:]}"] = b.get("tpr@fpr5")
    thr_idx = np.flatnonzero(keep)[is_threat]
    if len(thr_idx):
        pick = np.random.default_rng(0).choice(thr_idx, min(1000, len(thr_idx)), replace=False)
        up = AN.upsample_maps(st["maps"][pick], gt.shape[-1])
        row["pixel_auroc"] = EV.pixel_auroc(up, gt[pick])
    else:
        row["pixel_auroc"] = np.nan
    lo, hi = EV.auroc_ci(is_threat, st["score_max"][keep], n_boot=ctx.n_boot)
    row.update(bag_auroc_lo=lo, bag_auroc_hi=hi)
    info = load_json(ctx.runs / name / "bank.json")
    row.update(bank_mode=info["mode"], bank_size=info["bank_size"], coreset=info["coreset"], layers=info["layers"])
    return row


# ----------------------------------------------------------------------------- open set (RQ4)
def openset_eval(ctx, s1_open, cfg_open, bank_open, heldout_ids, s1_full=None, cfg_full=None):
    held = set(int(c) for c in heldout_ids)
    known = [c for c in range(NUM_CLASSES) if c not in held]
    dv, dt = part_df(ctx, "val"), part_df(ctx, "test")
    pv = predict_stage1(ctx, s1_open, cfg_open, "val")
    pt = predict_stage1(ctx, s1_open, cfg_open, "test")
    sv = score_bank(ctx, bank_open, "val")
    st = score_bank(ctx, bank_open, "test")
    fit = load_json(ctx.runs / bank_open / "null_fit.json")
    clean_v = np.array([len(l) == 0 for l in dv["labels"]])
    kn_v = CS.logit(pv["probs"][:, known].max(1))
    fit_known = AN.robust_fit(kn_v[clean_v]) if clean_v.sum() >= 5 else AN.robust_fit(kn_v)
    labs = [set(l) for l in dt["labels"]]
    U = np.array([bool(l) and l <= held for l in labs])
    K = np.array([bool(l) and not (l & held) for l in labs])
    B = np.array([not l for l in labs])
    s_known = AN.zscore(CS.logit(pt["probs"][:, known].max(1)), fit_known)
    s_anom = AN.zscore(st["score_max"], fit["img_fit"])
    s_fused = np.maximum(s_known, s_anom)
    scores = {"known-class score": s_known, "anomaly score": s_anom, "fused (max z)": s_fused}
    if s1_full is not None:
        pf = predict_stage1(ctx, s1_full, cfg_full, "test")
        scores["oracle: model trained on all classes"] = pf["probs"].max(1)
    rows = []
    for name, s in scores.items():
        for pop, mask in (("unseen threats", U), ("known threats", K)):
            sel = mask | B
            b = EV.bag_report(mask[sel], s[sel])
            rows.append(dict(score=name, population=pop, n_threat=int(mask.sum()), n_clean=int(B.sum()),
                             auroc=b.get("auroc"), tpr_at_fpr5=b.get("tpr@fpr5"), tpr_at_fpr1=b.get("tpr@fpr1")))
    return pd.DataFrame(rows), scores, dict(U=U, K=K, B=B)


# ----------------------------------------------------------------------------- stratified analysis
def strata_tables(ctx, preds: dict, thr_by_run: dict):
    dt = part_df(ctx, "test")
    tr = part_df(ctx, "train")
    size_edges = EV.terciles([v for vs in tr["rel_areas"] for v in vs])
    clut_edges = EV.terciles([v for vs in tr["clutter"] for v in vs])
    S_size = EV.pair_strata(dt, "rel_areas", size_edges)
    S_clut = EV.pair_strata(dt, "clutter", clut_edges)
    out = {}
    for name, P in preds.items():
        Y = label_matrix(dt)
        out[name] = {"object size": EV.stratified_report(Y, P, S_size, thr_by_run[name], ("small", "medium", "large")),
                     "clutter around threat": EV.stratified_report(Y, P, S_clut, thr_by_run[name])}
    return out, dict(size_edges=size_edges.tolist(), clutter_edges=clut_edges.tolist())


def caption_strata(dt):
    """Clutter level parsed from captions, if the captions mention it (else None)."""
    import re
    pat = re.compile(r"\b(limited|light|low|minimal|medium|moderate|heavy|heavily|high|extreme|extremely)\b"
                     r"[\w\s-]{0,15}clutter", re.I)
    level = {"limited": 0, "light": 0, "low": 0, "minimal": 0, "medium": 1, "moderate": 1, "heavy": 2,
             "heavily": 2, "high": 2, "extreme": 3, "extremely": 3}
    out = []
    for c in dt["caption"].fillna(""):
        m = pat.search(str(c))
        out.append(level[m.group(1).lower()] if m else -1)
    out = np.asarray(out)
    return out if (out >= 0).mean() > 0.3 else None


def latency_table(ctx, s1_name, s1_cfg, s2_name=None, s2_cfg=None, bank_name=None, n=50):
    device = get_device()
    rows = []
    m1 = load_run_model(s1_cfg, ctx.runs / s1_name, device)
    rows.append(dict(stage="Stage 1 (Look) network", ms=measure_latency(m1, s1_cfg["img_size"], device, n)))
    del m1
    if bank_name and (ctx.runs / bank_name / "bank.pt").exists():
        bank = torch.load(ctx.runs / bank_name / "bank.pt", weights_only=True).to(device).float()
        ext = AN.PatchFeatures(pretrained=not ctx.smoke).to(device)
        x = torch.randint(0, 255, (1, 3, ctx.cache_size, ctx.cache_size), dtype=torch.uint8, device=device)
        for _ in range(3):
            ext(x)
        cuda_sync()
        t = time.time()
        for _ in range(n):
            f = ext(x).float().reshape(-1, bank.shape[1])
            d2 = (f ** 2).sum(1, keepdim=True) + (bank ** 2).sum(1)[None] - 2 * f @ bank.T
            d2.min(1)
        cuda_sync()
        rows.append(dict(stage="Anomaly branch (features + NN search)", ms=(time.time() - t) / n * 1000))
    if s2_name:
        m2 = load_run_model(s2_cfg, ctx.runs / s2_name, device)
        rows.append(dict(stage="Stage 2 (Zoom), per crop", ms=measure_latency(m2, s2_cfg["img_size"], device, n)))
    return pd.DataFrame(rows)


__all__ = [n for n in dir() if not n.startswith("_")]
