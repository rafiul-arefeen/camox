"""Look -> Zoom cascade: proposals from Stage-1 maps, crop aggregation, score fusion, open-set fusion."""
from __future__ import annotations

import cv2
import numpy as np

from .config import NUM_CLASSES
from .evaluate import mean_ap


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _components(U, thr, min_cells):
    bw = (U >= thr).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    comps = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < min_cells:
            continue
        comps.append((float(U[lab == i].max()), x, y, x + w, y + h))
    return comps


def extract_proposals(seg_u8, meta, thr=0.35, topk=5, min_cells=2, anomaly_z=None, z0=3.0, cache_size=512,
                      fallback=0.05):
    """Connected components of the suspicion map -> boxes in ORIGINAL image coordinates.

    seg_u8: (h,w) uint8 threat probability (Stage-1 mask head) or None.
    anomaly_z: optional (h',w') z-scored anomaly map; cells become suspicious when z > z0.
    meta: dict(scale, pad_x, pad_y, img_w, img_h).
    If nothing passes `thr` but the map still peaks above `fallback`, the region around the peak
    (cells >= half the peak) is proposed, so Stage 2 always gets to look at the most suspicious spot.
    """
    maps = []
    if seg_u8 is not None:
        maps.append(seg_u8.astype(np.float32) / 255.0)
    if anomaly_z is not None:
        size = maps[0].shape if maps else (cache_size // 4, cache_size // 4)
        a = cv2.resize(_sigmoid(anomaly_z.astype(np.float32) - z0), (size[1], size[0]),
                       interpolation=cv2.INTER_LINEAR)
        maps.append(a)
    if not maps:
        return []
    U = np.maximum.reduce(maps) if len(maps) > 1 else maps[0]
    comps = _components(U, thr, min_cells)
    if not comps and fallback is not None and U.max() >= fallback:
        comps = _components(U, max(fallback, 0.5 * float(U.max())), 1)
    comps.sort(key=lambda t: -t[0])
    f = cache_size / U.shape[0]
    out = []
    for peak, x1, y1, x2, y2 in comps[:topk]:
        bx = np.array([x1 * f, y1 * f, x2 * f, y2 * f], np.float32)
        bx[[0, 2]] = (bx[[0, 2]] - meta["pad_x"]) / meta["scale"]
        bx[[1, 3]] = (bx[[1, 3]] - meta["pad_y"]) / meta["scale"]
        bx[[0, 2]] = bx[[0, 2]].clip(0, meta["img_w"])
        bx[[1, 3]] = bx[[1, 3]].clip(0, meta["img_h"])
        if bx[2] - bx[0] < 2 or bx[3] - bx[1] < 2:
            continue
        out.append([float(v) for v in bx] + [peak])
    return out


def proposals_for(df, seg, anomaly_z=None, **kw):
    """df rows aligned with seg[i] (and anomaly_z[i]) -> {uid: [[x1,y1,x2,y2,peak], ...]}"""
    props = {}
    for i, r in enumerate(df.itertuples(index=False)):
        meta = dict(scale=r.scale, pad_x=r.pad_x, pad_y=r.pad_y, img_w=r.img_w, img_h=r.img_h)
        props[r.uid] = extract_proposals(seg[i] if seg is not None else None, meta,
                                         anomaly_z=None if anomaly_z is None else anomaly_z[i], **kw)
    return props


def gt_boxes(df):
    return {r.uid: [list(b) for b in r.boxes if b[4] >= 0] for r in df.itertuples(index=False)}


def aggregate_crops(crop_probs, crop_uids, uids, topk=None, crop_k=None):
    """Max over an image's crops. Returns (N,C) scores and (N,) has-proposal flags."""
    pos = {u: i for i, u in enumerate(uids)}
    out = np.zeros((len(uids), crop_probs.shape[1] if len(crop_probs) else NUM_CLASSES), np.float32)
    has = np.zeros(len(uids), bool)
    for j, u in enumerate(crop_uids):
        if topk is not None and crop_k is not None and crop_k[j] >= topk:
            continue
        i = pos.get(u)
        if i is None:
            continue
        out[i] = np.maximum(out[i], crop_probs[j])
        has[i] = True
    return out, has


def fuse(p1, p2, has, alpha=0.5, rule="avg"):
    if rule == "s1":
        return p1.copy()
    if rule == "s2":
        out = p2.copy()
        out[~has] = 0.0
        return out
    if rule == "max":
        out = np.maximum(p1, p2)
    else:
        out = alpha * p1 + (1 - alpha) * p2
    out[~has] = p1[~has]
    return out


def tune_alpha(Y, p1, p2, has, grid=None, classes=None):
    grid = np.linspace(0, 1, 11) if grid is None else grid
    best = (-1, 0.5)
    cols = slice(None) if classes is None else classes
    for a in grid:
        m = mean_ap(Y[:, cols], fuse(p1, p2, has, a, "avg")[:, cols])
        if m > best[0]:
            best = (m, float(a))
    return best[1], best[0]


def box_predictions(crop_df, crop_probs, p_img=None, uids=None, topc=3, min_score=0.02, mix=0.0):
    """Class-aware boxes: each proposal box gets its top classes from the Stage-2 crop scores.
    mix>0 blends in the image-level fused score: s = (1-mix)*p_crop + mix*p_img."""
    pos = {u: i for i, u in enumerate(uids)} if uids is not None else {}
    preds = {}
    for j, r in enumerate(crop_df.itertuples(index=False)):
        p = crop_probs[j]
        if mix > 0 and p_img is not None and r.uid in pos:
            p = (1 - mix) * p + mix * p_img[pos[r.uid]]
        top = np.argsort(-p)[:topc]
        for c in top:
            if p[c] >= min_score:
                preds.setdefault(r.uid, []).append(list(r.box) + [int(c), float(p[c])])
    return preds


def s1_box_predictions(props, p1, uids, topc=3, min_score=0.02):
    """Stage-1-only boxes: proposal box x the image's top classes (score = p_c * peak)."""
    pos = {u: i for i, u in enumerate(uids)}
    preds = {}
    for u, boxes in props.items():
        if u not in pos:
            continue
        p = p1[pos[u]]
        top = np.argsort(-p)[:topc]
        for b in boxes:
            for c in top:
                s = float(p[c] * b[4])
                if s >= min_score:
                    preds.setdefault(u, []).append(list(b[:4]) + [int(c), s])
    return preds


def logit(p, eps=1e-6):
    p = np.clip(np.asarray(p, np.float64), eps, 1 - eps)
    return np.log(p / (1 - p))


__all__ = ["extract_proposals", "proposals_for", "gt_boxes", "aggregate_crops", "fuse", "tune_alpha",
           "box_predictions", "s1_box_predictions", "logit"]
