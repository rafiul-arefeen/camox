"""Metrics: multi-label recognition, bag-level screening, localisation, box AP, bootstrap CIs."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from .config import CLASS_NAMES, NUM_CLASSES
from .utils import box_iou

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


# ----------------------------------------------------------------------------- multi-label
def per_class_ap(Y, P, min_pos=1):
    ap = np.full(Y.shape[1], np.nan)
    for c in range(Y.shape[1]):
        if Y[:, c].sum() >= min_pos and Y[:, c].sum() < len(Y):
            ap[c] = average_precision_score(Y[:, c], P[:, c])
    return ap


def per_class_auroc(Y, P):
    au = np.full(Y.shape[1], np.nan)
    for c in range(Y.shape[1]):
        if 0 < Y[:, c].sum() < len(Y):
            au[c] = roc_auc_score(Y[:, c], P[:, c])
    return au


def mean_ap(Y, P):
    return float(np.nanmean(per_class_ap(Y, P))) if len(Y) else float("nan")


def tune_thresholds(Y, P, grid=None):
    """Per-class threshold maximising F1 on (validation) data."""
    grid = np.linspace(0.02, 0.98, 49) if grid is None else grid
    thr = np.full(Y.shape[1], 0.5, np.float32)
    for c in range(Y.shape[1]):
        if Y[:, c].sum() == 0:
            continue
        pred = P[:, c][None, :] >= grid[:, None]
        tp = (pred & (Y[:, c][None, :] > 0)).sum(1)
        fp = (pred & (Y[:, c][None, :] == 0)).sum(1)
        fn = ((~pred) & (Y[:, c][None, :] > 0)).sum(1)
        f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
        thr[c] = grid[int(np.argmax(f1))]
    return thr


def thresholds_for_recall(Y, P, recall=0.95):
    thr = np.full(Y.shape[1], 0.5, np.float32)
    for c in range(Y.shape[1]):
        pos = np.sort(P[Y[:, c] > 0, c])
        if len(pos):
            k = int(np.floor((1 - recall) * len(pos)))
            thr[c] = pos[min(k, len(pos) - 1)]
    return thr


def f1_scores(Y, P, thr):
    pred = P >= thr[None, :]
    tp = (pred & (Y > 0)).sum(0)
    fp = (pred & (Y == 0)).sum(0)
    fn = ((~pred) & (Y > 0)).sum(0)
    valid = Y.sum(0) > 0
    f1_c = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    macro = float(f1_c[valid].mean()) if valid.any() else float("nan")
    micro = float(2 * tp.sum() / max(2 * tp.sum() + fp.sum() + fn.sum(), 1))
    prec = tp.sum() / max(tp.sum() + fp.sum(), 1)
    rec = tp.sum() / max(tp.sum() + fn.sum(), 1)
    return dict(macro_f1=macro, micro_f1=micro, micro_precision=float(prec), micro_recall=float(rec),
                per_class_f1=f1_c)


def multilabel_report(Y, P, thr=None, classes=None):
    classes = classes if classes is not None else list(range(Y.shape[1]))
    Yc, Pc = Y[:, classes], P[:, classes]
    ap = per_class_ap(Yc, Pc)
    au = per_class_auroc(Yc, Pc)
    out = dict(mAP=float(np.nanmean(ap)), macro_auroc=float(np.nanmean(au)), per_class_ap=ap, per_class_auroc=au)
    if thr is not None:
        out.update(f1_scores(Yc, Pc, np.asarray(thr)[classes]))
    return out


# ----------------------------------------------------------------------------- bag level
def bag_scores(P, classes=None):
    return P.max(1) if classes is None else P[:, classes].max(1)


def bag_report(is_threat, score, recalls=(0.95, 0.99), fprs=(0.05, 0.01)):
    is_threat = np.asarray(is_threat).astype(bool)
    score = np.asarray(score, np.float64)
    out = dict(n_threat=int(is_threat.sum()), n_clean=int((~is_threat).sum()))
    if is_threat.all() or (~is_threat).all():
        out["auroc"] = float("nan")
        return out
    out["auroc"] = float(roc_auc_score(is_threat, score))
    fpr, tpr, _ = roc_curve(is_threat, score)
    for f in fprs:
        out[f"tpr@fpr{int(f * 100)}"] = float(np.interp(f, fpr, tpr))
    pos = np.sort(score[is_threat])
    neg = score[~is_threat]
    for r in recalls:
        k = int(np.floor((1 - r) * len(pos)))
        t = pos[min(k, len(pos) - 1)]
        out[f"clear@rec{int(r * 100)}"] = float((neg < t).mean())
    return out


# ----------------------------------------------------------------------------- bootstrap
def bootstrap_ci(fn, n, n_boot=1000, seed=0, alpha=0.05):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            v = fn(idx)
        except Exception:
            continue
        if v is not None and np.isfinite(v):
            vals.append(v)
    if not vals:
        return float("nan"), float("nan")
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


def _sorted_class(y, p):
    order = np.argsort(-p, kind="mergesort")
    ps = p[order]
    ends = np.r_[np.flatnonzero(np.diff(ps)), len(ps) - 1]   # last index of every tied-score group
    return order, ends


def _weighted_ap(ys, W, ends):
    """AP (sklearn definition, ties grouped) for many bootstrap weight vectors at once.
    ys: (N,) labels in score order; W: (R,N) resample counts in score order."""
    tp = np.cumsum(W * ys[None, :], axis=1)[:, ends]
    cnt = np.cumsum(W, axis=1)[:, ends]
    P = tp[:, -1]
    prec = tp / np.maximum(cnt, 1e-12)
    dtp = np.diff(np.concatenate([np.zeros((len(W), 1)), tp], axis=1), axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        ap = (dtp * prec).sum(1) / P
    ap[P <= 0] = np.nan
    ap[P >= cnt[:, -1]] = np.nan  # no negatives in the resample
    return ap


def bootstrap_maps(Y, P_list, n_boot=1000, seed=0, classes=None, chunk=50):
    """Paired bootstrap: (n_boot, len(P_list)) array of mAP values on the same resamples."""
    Y = np.asarray(Y)
    n = len(Y)
    cols = range(Y.shape[1]) if classes is None else classes
    rng = np.random.default_rng(seed)
    out = np.full((n_boot, len(P_list)), np.nan)
    prep = []
    for c in cols:
        if Y[:, c].sum() == 0:
            continue
        prep.append([(c,) + _sorted_class(Y[:, c], np.asarray(P)[:, c]) for P in P_list])
    for s in range(0, n_boot, chunk):
        r = min(chunk, n_boot - s)
        W = np.stack([np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(r)]).astype(np.float64)
        per = np.full((r, len(P_list), len(prep)), np.nan)
        for j, items in enumerate(prep):
            for m, (c, order, ends) in enumerate(items):
                per[:, m, j] = _weighted_ap(Y[order, c].astype(np.float64), W[:, order], ends)
        with np.errstate(all="ignore"):
            out[s:s + r] = np.nanmean(per, axis=2)
    return out


def map_ci(Y, P, n_boot=1000, seed=0, classes=None):
    v = bootstrap_maps(Y, [P], n_boot, seed, classes)[:, 0]
    v = v[np.isfinite(v)]
    if not len(v):
        return float("nan"), float("nan")
    return float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))


def auroc_ci(is_threat, score, n_boot=1000, seed=0):
    t, s = np.asarray(is_threat), np.asarray(score)

    def f(i):
        if t[i].all() or (~t[i].astype(bool)).all():
            return None
        return roc_auc_score(t[i], s[i])
    return bootstrap_ci(f, len(t), n_boot, seed)


def paired_delta(Y, P_a, P_b, n_boot=1000, seed=0, classes=None):
    """mAP(b) - mAP(a) with a paired bootstrap CI and the share of resamples where b <= a."""
    cols = list(range(Y.shape[1])) if classes is None else classes
    base = mean_ap(Y[:, cols], P_b[:, cols]) - mean_ap(Y[:, cols], P_a[:, cols])
    v = bootstrap_maps(Y, [P_a, P_b], n_boot, seed, classes)
    d = v[:, 1] - v[:, 0]
    d = d[np.isfinite(d)]
    return dict(delta=float(base), lo=float(np.quantile(d, 0.025)), hi=float(np.quantile(d, 0.975)),
                p_le0=float((d <= 0).mean()))


# ----------------------------------------------------------------------------- localisation
def seg_report(pred, gt, is_threat, thr=0.5, chunk=1000):
    """pred, gt: (N,h,w) uint8 (0..255) or float (0..1). Dice / IoU over threat scans (per-image mean).
    Works chunk-wise on uint8 data so a 16k-scan test set needs little RAM."""
    def binar(a, t):
        return a >= (t * 255 if a.dtype == np.uint8 else t)
    sel = np.flatnonzero(np.asarray(is_threat).astype(bool))
    dices, ious = [], []
    for s in range(0, len(sel), chunk):
        idx = sel[s:s + chunk]
        pb = binar(pred[idx], thr)
        gb = binar(gt[idx], 0.5)
        inter = (pb & gb).sum((1, 2))
        union = (pb | gb).sum((1, 2))
        tot = pb.sum((1, 2)) + gb.sum((1, 2))
        valid = gb.sum((1, 2)) > 0
        dices.append(np.where(tot > 0, 2 * inter / np.maximum(tot, 1), 1.0)[valid])
        ious.append(np.where(union > 0, inter / np.maximum(union, 1), 1.0)[valid])
    dice = np.concatenate(dices) if dices else np.zeros(0)
    iou = np.concatenate(ious) if ious else np.zeros(0)
    clean = np.flatnonzero(~np.asarray(is_threat).astype(bool))
    fp_clean = float(binar(pred[clean], thr).mean()) if len(clean) else float("nan")
    return dict(dice=float(dice.mean()) if len(dice) else float("nan"),
                iou=float(iou.mean()) if len(iou) else float("nan"),
                clean_pixel_fp=fp_clean, n=int(len(dice)))


def pixel_auroc(maps, gts, max_pixels=2_000_000, seed=0):
    rng = np.random.default_rng(seed)
    m = maps.reshape(-1).astype(np.float32)
    g = (gts.reshape(-1).astype(np.float32) > (127 if gts.max() > 1.5 else 0.5))
    if len(m) > max_pixels:
        idx = rng.choice(len(m), max_pixels, replace=False)
        m, g = m[idx], g[idx]
    if g.all() or (~g).all():
        return float("nan")
    return float(roc_auc_score(g, m))


def pointing_game(cam_peaks, grid, boxes_cache, labels, img_size, tol=15.0):
    """cam_peaks: (N,C,2) argmax (row, col) on a grid x grid CAM. boxes in cache (img_size) coordinates."""
    hits = np.zeros(NUM_CLASSES)
    tot = np.zeros(NUM_CLASSES)
    cell = img_size / grid
    for i, (labs, boxes) in enumerate(zip(labels, boxes_cache)):
        if not labs or not boxes:
            continue
        b = np.asarray(boxes, np.float32).reshape(-1, 5)
        for c in labs:
            bc = b[b[:, 4] == c]
            if len(bc) == 0:
                continue
            r, q = cam_peaks[i, c]
            y, x = (r + 0.5) * cell, (q + 0.5) * cell
            ok = ((x >= bc[:, 0] - tol) & (x <= bc[:, 2] + tol) & (y >= bc[:, 1] - tol) & (y <= bc[:, 3] + tol)).any()
            hits[c] += ok
            tot[c] += 1
    per = np.where(tot > 0, hits / np.maximum(tot, 1), np.nan)
    return dict(pointing_acc=float(hits.sum() / max(tot.sum(), 1)), pointing_macro=float(np.nanmean(per)),
                per_class=per)


def voc_ap(rec, prec):
    mrec = np.concatenate([[0.0], rec, [1.0]])
    mpre = np.concatenate([[0.0], prec, [0.0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def box_ap(preds, gts, iou_thr=0.5, classes=None):
    """preds: uid -> [[x1,y1,x2,y2,cls,score],...]; gts: uid -> [[x1,y1,x2,y2,cls],...] (same coordinates)."""
    classes = range(NUM_CLASSES) if classes is None else classes
    aps = np.full(NUM_CLASSES, np.nan)
    recs = np.full(NUM_CLASSES, np.nan)
    for c in classes:
        gt_c = {u: np.asarray([g[:4] for g in gl if int(g[4]) == c], np.float32).reshape(-1, 4)
                for u, gl in gts.items()}
        npos = sum(len(v) for v in gt_c.values())
        if npos == 0:
            continue
        dets = [(u, p[5], p[:4]) for u, pl in preds.items() for p in pl if int(p[4]) == c]
        if not dets:
            aps[c], recs[c] = 0.0, 0.0
            continue
        dets.sort(key=lambda t: -t[1])
        used = {u: np.zeros(len(v), bool) for u, v in gt_c.items()}
        tp = np.zeros(len(dets))
        for k, (u, _, b) in enumerate(dets):
            g = gt_c.get(u)
            if g is None or len(g) == 0:
                continue
            ious = box_iou(np.asarray([b], np.float32), g)[0]
            j = int(np.argmax(ious))
            if ious[j] >= iou_thr and not used[u][j]:
                used[u][j] = True
                tp[k] = 1
        ctp = np.cumsum(tp)
        rec = ctp / npos
        prec = ctp / np.arange(1, len(dets) + 1)
        aps[c] = voc_ap(rec, prec)
        recs[c] = rec[-1]
    return dict(box_mAP50=float(np.nanmean(aps)), per_class_ap=aps, recall=float(np.nanmean(recs)))


def proposal_recall(props, gts, iou_thr=0.5, ioa_thr=0.5):
    """Share of GT boxes covered by at least one proposal (IoU>=iou_thr or GT-IoA>=ioa_thr)."""
    from .utils import box_ioa
    hit = tot = hit_iou = 0
    for u, gl in gts.items():
        g = np.asarray([x[:4] for x in gl], np.float32).reshape(-1, 4)
        if len(g) == 0:
            continue
        tot += len(g)
        p = np.asarray([x[:4] for x in props.get(u, [])], np.float32).reshape(-1, 4)
        if len(p) == 0:
            continue
        iou = box_iou(g, p)
        ioa = box_ioa(g, p)
        hit_iou += int((iou.max(1) >= iou_thr).sum())
        hit += int(((iou.max(1) >= iou_thr) | (ioa.max(1) >= ioa_thr)).sum())
    return dict(prop_recall_iou50=hit_iou / max(tot, 1), prop_recall_cover=hit / max(tot, 1), n_gt=tot)


# ----------------------------------------------------------------------------- stratified analysis
def terciles(values):
    v = np.asarray([x for x in values if np.isfinite(x)])
    if len(v) < 3:
        return np.array([np.inf, np.inf])
    return np.quantile(v, [1 / 3, 2 / 3])


def pair_strata(df, key, edges):
    """(N,C) stratum per positive (image, class) pair using the largest instance of that class."""
    S = -np.ones((len(df), NUM_CLASSES), np.int8)
    for i, (labs, boxes, vals) in enumerate(zip(df["labels"], df["boxes"], df[key])):
        if not labs or not isinstance(vals, (list, tuple)) or not boxes:
            continue
        b = np.asarray(boxes, np.float32).reshape(-1, 5)
        v = np.asarray(vals, np.float32)
        for c in labs:
            sel = (b[:, 4] == c) & np.isfinite(v)
            if sel.any():
                x = v[sel].max()
                S[i, c] = int(np.searchsorted(edges, x, side="right"))
    return S


def stratified_report(Y, P, S, thr, names=("low", "mid", "high")):
    rows = []
    for s, name in enumerate(names):
        aps, recs, npos = [], [], 0
        for c in range(Y.shape[1]):
            pos = (S[:, c] == s)
            neg = Y[:, c] == 0
            if pos.sum() < 3:
                continue
            sel = pos | neg
            aps.append(average_precision_score(pos[sel], P[sel, c]))
            recs.append(float((P[pos, c] >= thr[c]).mean()))
            npos += int(pos.sum())
        rows.append(dict(stratum=name, mAP=float(np.mean(aps)) if aps else np.nan,
                         recall=float(np.mean(recs)) if recs else np.nan, positives=npos))
    return pd.DataFrame(rows)


def confusion_single(Y, P, classes=None):
    """Confusion matrix on single-label threat scans (argmax prediction)."""
    single = Y.sum(1) == 1
    t = Y[single].argmax(1)
    p = P[single].argmax(1)
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), int)
    np.add.at(cm, (t, p), 1)
    return cm


def results_row(name, Y, P, thr, is_threat, extra=None, n_boot=500, classes=None, seed=0):
    rep = multilabel_report(Y, P, thr, classes)
    bs = bag_scores(P, classes)
    bag = bag_report(is_threat, bs)
    lo, hi = map_ci(Y, P, n_boot, seed, classes) if n_boot else (np.nan, np.nan)
    blo, bhi = auroc_ci(is_threat, bs, n_boot, seed) if n_boot else (np.nan, np.nan)
    row = dict(run=name, mAP=rep["mAP"], mAP_lo=lo, mAP_hi=hi, macro_auroc=rep["macro_auroc"],
               macro_f1=rep.get("macro_f1"), micro_f1=rep.get("micro_f1"), bag_auroc=bag.get("auroc"),
               bag_auroc_lo=blo, bag_auroc_hi=bhi,
               tpr_at_fpr5=bag.get("tpr@fpr5"), clear_at_rec95=bag.get("clear@rec95"),
               clear_at_rec99=bag.get("clear@rec99"))
    if extra:
        row.update(extra)
    return row, rep


__all__ = [n for n in dir() if not n.startswith("_")] + ["CLASS_NAMES"]
