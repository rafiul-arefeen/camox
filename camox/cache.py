"""One-time caches: letterboxed scans + soft masks, TIP donor bank, Stage-2 zoom patches."""
from __future__ import annotations

import math
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .config import NUM_CLASSES
from .indexing import rasterize_annotations
from .material import local_clutter_score
from .splits import image_signature
from .utils import (box_ioa, boxes_orig_to_cache, crop_with_pad, imread_any, imread_rgb, imwrite_any,
                    imwrite_rgb, letterbox, square_region)

PATCH_SIZE = 320          # stored zoom patch side (pixels)
PATCH_CONTEXT = 2.0       # stored patch covers 2x the box size; crops take the centre part


def _progress(it, total, desc):
    try:
        from tqdm.auto import tqdm
        return tqdm(it, total=total, desc=desc)
    except Exception:
        return it


def cache_paths(cache_dir, size):
    cache_dir = Path(cache_dir)
    return cache_dir / f"img{size}", cache_dir / f"mask{size}"


def _read_seg_png(path, w, h):
    m = imread_any(path, cv2.IMREAD_UNCHANGED)
    if m is None:
        return None
    if m.ndim == 3:
        if m.shape[2] == 4:
            alpha = m[..., 3]
            if alpha.min() != alpha.max():
                m = alpha
            else:
                m = m[..., :3]
    if m.ndim == 3:
        flat = m.reshape(-1, m.shape[2])
        vals, counts = np.unique(flat[:: max(1, len(flat) // 20000)], axis=0, return_counts=True)
        bg = vals[counts.argmax()].astype(np.int16)
        fg = (np.abs(m.astype(np.int16) - bg[None, None]).max(2) > 10)
    else:
        vals, counts = np.unique(m.reshape(-1)[:: max(1, m.size // 20000)], return_counts=True)
        bg = vals[counts.argmax()]
        fg = m != bg
    fg = fg.astype(np.uint8)
    if w and h and (fg.shape[1] != int(w) or fg.shape[0] != int(h)):
        fg = cv2.resize(fg, (int(w), int(h)), interpolation=cv2.INTER_NEAREST)
    return fg


def _box_mask(boxes, w, h):
    m = np.zeros((int(h), int(w)), np.uint8)
    for b in boxes:
        x1, y1, x2, y2 = [int(round(v)) for v in b[:4]]
        m[max(0, y1):max(0, y2), max(0, x1):max(0, x2)] = 1
    return m


def _valid_num(v):
    try:
        return v is not None and not math.isnan(float(v)) and float(v) > 0
    except (TypeError, ValueError):
        return False


def _cache_one(args):
    row, size, img_dir, mask_dir, overwrite, bbox_format = args
    out = dict(uid=row["uid"])
    ip, mp = img_dir / f"{row['uid']}.jpg", mask_dir / f"{row['uid']}.png"
    rgb = None
    need = overwrite or not ip.exists() or not mp.exists() or not _valid_num(row.get("img_w")) \
        or not _valid_num(row.get("img_h"))
    if need:
        rgb = imread_rgb(row["img_path"])
        if rgb is None:
            out["cache_ok"] = False
            return out
        h, w = rgb.shape[:2]
        lb, scale, px, py = letterbox(rgb, size, 255)
        imwrite_rgb(ip, lb, quality=95)
        # ---------------- mask (soft coverage in 0..255)
        fg, src = None, row["mask_source"]
        if src == "png" and row.get("seg_path"):
            fg = _read_seg_png(row["seg_path"], w, h)
            if fg is None:
                src = "poly" if row.get("json_path") else "box"
        if fg is None and src == "poly" and row.get("json_path"):
            fg, n = rasterize_annotations(row["json_path"], w, h, fmt=bbox_format)
            if n == 0:
                fg, src = None, "box"
        if fg is None and src == "box" and len(row["boxes"]):
            fg = _box_mask(row["boxes"], w, h)
        if fg is None:
            fg = np.zeros((h, w), np.uint8)
            src = "none"
        if row["folder_id"] == 22:  # benign bag: ignore stray pixels
            fg[:] = 0
        m_lb, _, _, _ = letterbox(fg.astype(np.float32), size, 0.0, interp=cv2.INTER_AREA)
        imwrite_any(mp, np.clip(m_lb * 255.0, 0, 255).astype(np.uint8))
        if src == "png" and len(row["boxes"]) and m_lb.sum() > 0:
            # share of mask pixels that fall inside the parsed boxes: low values reveal a box-format problem
            inside = np.zeros(m_lb.shape, bool)
            for b in boxes_orig_to_cache(row["boxes"], scale, px, py):
                inside[max(0, int(b[1]) - 1):int(math.ceil(b[3])) + 1, max(0, int(b[0]) - 1):int(math.ceil(b[2])) + 1] = True
            out["box_mask_agree"] = float((m_lb * inside).sum() / m_lb.sum())
        out.update(cache_ok=True, scale=scale, pad_x=px, pad_y=py, img_w=w, img_h=h, mask_used=src,
                   mask_frac=float((fg > 0).mean()))
    else:
        lb = imread_rgb(ip)
        w, h = row["img_w"], row["img_h"]
        scale = size / max(w, h)
        nw, nh = int(round(w * scale)), int(round(h * scale))
        out.update(cache_ok=True, scale=scale, pad_x=(size - nw) // 2, pad_y=(size - nh) // 2, img_w=w, img_h=h,
                   mask_used=row["mask_source"], mask_frac=np.nan)
    sig = image_signature(lb)
    out["sig_d"], out["sig_a"] = sig
    # ---------------- per-object geometry / clutter statistics (cache coordinates)
    bc = boxes_orig_to_cache(row["boxes"], out["scale"], out["pad_x"], out["pad_y"]) if len(row["boxes"]) else \
        np.zeros((0, 5), np.float32)
    out["boxes_cache"] = bc.tolist()
    rel = [float((b[2] - b[0]) * (b[3] - b[1]) / max(1.0, out["img_w"] * out["img_h"])) for b in row["boxes"]]
    out["rel_areas"] = rel
    clut = [local_clutter_score(lb, b) for b in bc]
    out["clutter"] = [c[0] for c in clut]
    out["dens_box"] = [c[1] for c in clut]
    v = lb.max(2)
    out["bag_frac"] = float((v < 235).mean())
    return out


def cache_progress_file(cache_dir, size):
    return Path(cache_dir) / f"_cache_progress_{size}.pkl"


def build_cache(df, cache_dir, size=512, workers=8, overwrite=False, bbox_format="xyxy", chunk=2000):
    """Letterbox every scan (white padding) + soft threat masks; returns df with cache columns.

    Resumable: finished scans are recorded every `chunk` scans, so stopping the notebook during this step
    only repeats the last unfinished chunk next time."""
    img_dir, mask_dir = cache_paths(cache_dir, size)
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    prog = cache_progress_file(cache_dir, size)
    done = {}
    if prog.exists() and not overwrite:
        try:
            done = {r["uid"]: r for r in pd.read_pickle(prog)}
        except Exception:
            done = {}
    rows = df.to_dict("records")
    wanted = {r["uid"] for r in rows}
    res = [v for k, v in done.items() if k in wanted]
    todo = [r for r in rows if r["uid"] not in done]
    if res:
        print(f"resuming: {len(res)} of {len(rows)} scans were cached in an earlier session")
    try:
        from tqdm.auto import tqdm
        bar = tqdm(total=len(rows), initial=len(res), desc=f"cache {size}px")
    except Exception:
        bar = None
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for s in range(0, len(todo), chunk):
            # files of unrecorded scans may be from an interrupted run, so they are always rewritten
            args = [(r, size, img_dir, mask_dir, True, bbox_format) for r in todo[s:s + chunk]]
            for out in ex.map(_cache_one, args):
                res.append(out)
                if bar is not None:
                    bar.update(1)
            tmp = prog.with_name(prog.name + ".tmp")
            pd.to_pickle(res, tmp)
            os.replace(tmp, prog)
    if bar is not None:
        bar.close()
    cdf = pd.DataFrame(res)
    keep = [c for c in cdf.columns if c != "uid"]
    out = df.drop(columns=[c for c in keep if c in df.columns]).merge(cdf, on="uid", how="left")
    return out


def load_cached(uid, cache_dir, size):
    img_dir, mask_dir = cache_paths(cache_dir, size)
    img = imread_rgb(img_dir / f"{uid}.jpg")
    m = imread_any(mask_dir / f"{uid}.png", cv2.IMREAD_GRAYSCALE)
    return img, m


# ----------------------------------------------------------------------------- TIP donors
def build_donors(df_train, cache_dir, size=512, max_per_class=1500, min_pixels=40, max_side_frac=0.6,
                 seed=0, workers=8):
    """Cut threat instances (RGB transmission + alpha) out of training scans for CA-TIP."""
    rng = np.random.default_rng(seed)
    ddir = Path(cache_dir) / f"donors{size}"
    ddir.mkdir(parents=True, exist_ok=True)
    cand = []
    for r in df_train.itertuples(index=False):
        if r.mask_used not in ("png", "poly") or not isinstance(r.boxes_cache, list) or not r.boxes_cache:
            continue
        bc = np.asarray(r.boxes_cache, np.float32).reshape(-1, 5)
        for k in range(len(bc)):
            if bc[k, 4] < 0:
                continue
            cand.append((r.uid, k, int(bc[k, 4])))
    by_cls = {}
    for c in cand:
        by_cls.setdefault(c[2], []).append(c)
    chosen = []
    for cls, lst in by_cls.items():
        idx = rng.permutation(len(lst))[:max_per_class]
        chosen.extend(lst[i] for i in idx)
    uid2row = {r.uid: r for r in df_train.itertuples(index=False)}

    def work(item):
        uid, k, cls = item
        r = uid2row[uid]
        img, m = load_cached(uid, cache_dir, size)
        if img is None or m is None:
            return None
        bc = np.asarray(r.boxes_cache, np.float32).reshape(-1, 5)
        b = bc[k]
        others = np.delete(bc[:, :4], k, 0)
        if len(others) and box_ioa(b[None, :4], others).max() > 0.3:
            return None
        x1, y1 = int(max(0, math.floor(b[0]) - 3)), int(max(0, math.floor(b[1]) - 3))
        x2, y2 = int(min(size, math.ceil(b[2]) + 3)), int(min(size, math.ceil(b[3]) + 3))
        if x2 - x1 < 6 or y2 - y1 < 6 or max(x2 - x1, y2 - y1) > max_side_frac * size:
            return None
        crop = img[y1:y2, x1:x2].astype(np.float32) / 255.0
        alpha = m[y1:y2, x1:x2].astype(np.float32) / 255.0
        # keep only mask pixels that belong to this box
        inside = np.zeros_like(alpha)
        inside[int(b[1]) - y1:int(math.ceil(b[3])) - y1, int(b[0]) - x1:int(math.ceil(b[2])) - x1] = 1
        alpha = alpha * inside
        if (alpha > 0.5).sum() < min_pixels:
            return None
        outside = alpha < 0.05
        if outside.sum() >= 20:
            bg = np.quantile(crop[outside], 0.9, axis=0)
        else:
            bg = np.array([1.0, 1.0, 1.0], np.float32)
        bg = np.clip(bg, 0.5, 1.0)
        trans = np.clip(crop / bg[None, None], 0, 1)
        rgba = np.dstack([trans, alpha])
        path = ddir / f"{cls:02d}" / f"{uid}_{k}.png"
        bgra = cv2.cvtColor((rgba * 255).round().astype(np.uint8), cv2.COLOR_RGBA2BGRA)
        imwrite_any(path, bgra)
        return dict(path=str(path), cls=cls, uid=uid, h=y2 - y1, w=x2 - x1)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        res = list(_progress(ex.map(work, chosen), len(chosen), "TIP donors"))
    return pd.DataFrame([r for r in res if r is not None])


# ----------------------------------------------------------------------------- Stage-2 patches
def _save_patch(img, region, path, boxes, sub_box):
    patch = crop_with_pad(img, region, PATCH_SIZE, 255)
    imwrite_rgb(path, patch, quality=92)
    x1, y1, x2, y2 = region
    s = PATCH_SIZE / (x2 - x1)
    rel = []
    for b in boxes:
        bx = [(b[0] - x1) * s, (b[1] - y1) * s, (b[2] - x1) * s, (b[3] - y1) * s]
        if bx[2] <= 0 or bx[3] <= 0 or bx[0] >= PATCH_SIZE or bx[1] >= PATCH_SIZE:
            continue
        rel.append(bx + [int(b[4])])
    return rel


def sample_background_boxes(img_shape, gt_boxes, sizes, n, rng, bag_mask=None, max_ioa=0.1):
    h, w = img_shape[:2]
    out = []
    tries = 0
    while len(out) < n and tries < n * 30:
        tries += 1
        s = float(rng.choice(sizes)) if len(sizes) else 0.1 * max(h, w)
        s = min(s, 0.8 * min(h, w))
        cx, cy = rng.uniform(s / 2, w - s / 2), rng.uniform(s / 2, h - s / 2)
        if bag_mask is not None:
            yy = int(np.clip(cy / h * bag_mask.shape[0], 0, bag_mask.shape[0] - 1))
            xx = int(np.clip(cx / w * bag_mask.shape[1], 0, bag_mask.shape[1] - 1))
            if not bag_mask[yy, xx]:
                continue
        b = [cx - s / 2, cy - s / 2, cx + s / 2, cy + s / 2]
        if len(gt_boxes) and (box_ioa(np.asarray(gt_boxes)[:, :4], np.asarray([b])).max() > max_ioa
                              or box_ioa(np.asarray([b]), np.asarray(gt_boxes)[:, :4]).max() > max_ioa):
            continue
        out.append(b)
    return out


def build_patches(records, out_dir, kind, proposals=None, n_random=1, sizes=None, seed=0, workers=8,
                  min_side=48.0, max_hard=3):
    """Zoom patches cut from the ORIGINAL scans.

    kind='train': GT boxes (positives) + random background + hard negatives from `proposals`.
    kind='eval' : one patch per proposal (proposals: uid -> list of [x1,y1,x2,y2,score]).
    kind='gt'   : one patch per GT box (oracle Look).
    Returns a DataFrame(path, uid, kind, boxes) where boxes are GT boxes in patch coordinates.
    """
    out_dir = Path(out_dir)
    rng_master = np.random.default_rng(seed)
    seeds = rng_master.integers(0, 2 ** 31, len(records))
    sizes = np.asarray(sizes if sizes is not None else [96.0])

    def work(i):
        r = records[i]
        rng = np.random.default_rng(int(seeds[i]))
        img = imread_rgb(r["img_path"])
        if img is None:
            return []
        gt = [b for b in r["boxes"] if b[4] >= 0] if r["boxes"] else []
        rows = []
        if kind in ("train", "gt"):
            for k, b in enumerate(gt):
                reg = square_region(b, PATCH_CONTEXT, min_side * PATCH_CONTEXT)
                p = out_dir / f"{r['uid']}_p{k}.jpg"
                rel = _save_patch(img, reg, p, gt, b)
                rows.append(dict(path=str(p), uid=r["uid"], kind="pos", k=k, boxes=rel, score=1.0,
                                 region=list(reg), box=[float(t) for t in b[:4]]))
        if kind == "train":
            small = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
            bag = small.max(2) < 235
            for k, b in enumerate(sample_background_boxes(img.shape, gt, sizes, n_random, rng, bag)):
                reg = square_region(b, PATCH_CONTEXT, min_side * PATCH_CONTEXT)
                p = out_dir / f"{r['uid']}_r{k}.jpg"
                rel = _save_patch(img, reg, p, gt, b)
                rows.append(dict(path=str(p), uid=r["uid"], kind="rand", k=k, boxes=rel, score=0.0,
                                 region=list(reg)))
            props = (proposals or {}).get(r["uid"], [])
            nh = 0
            for k, b in enumerate(props):
                if nh >= max_hard:
                    break
                if gt and box_ioa(np.asarray([b[:4]]), np.asarray(gt)[:, :4]).max() > 0.1:
                    continue
                reg = square_region(b, PATCH_CONTEXT, min_side * PATCH_CONTEXT)
                p = out_dir / f"{r['uid']}_h{k}.jpg"
                rel = _save_patch(img, reg, p, gt, b)
                rows.append(dict(path=str(p), uid=r["uid"], kind="hard", k=k, boxes=rel, score=float(b[4]),
                                 region=list(reg)))
                nh += 1
        if kind == "eval":
            for k, b in enumerate((proposals or {}).get(r["uid"], [])):
                reg = square_region(b, PATCH_CONTEXT, min_side * PATCH_CONTEXT)
                p = out_dir / f"{r['uid']}_e{k}.jpg"
                rel = _save_patch(img, reg, p, gt, b)
                rows.append(dict(path=str(p), uid=r["uid"], kind="prop", k=k, boxes=rel, score=float(b[4]),
                                 region=list(reg), box=[float(t) for t in b[:4]]))
        return rows

    out_dir.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        res = list(_progress(ex.map(work, range(len(records))), len(records), f"patches ({kind})"))
    flat = [row for rows in res for row in rows]
    return pd.DataFrame(flat)


def box_size_pool(df, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    sizes = [max(b[2] - b[0], b[3] - b[1]) for bs in df["boxes"] for b in bs]
    sizes = np.asarray(sizes, np.float32)
    if len(sizes) > n:
        sizes = rng.choice(sizes, n, replace=False)
    return sizes


__all__ = ["build_cache", "cache_progress_file", "load_cached", "build_donors", "build_patches", "box_size_pool",
           "cache_paths",
           "PATCH_SIZE", "PATCH_CONTEXT", "NUM_CLASSES"]
