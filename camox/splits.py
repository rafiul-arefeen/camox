"""Leakage-aware splitting: perceptual-hash grouping + grouped multi-label stratification."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CLASS_NAMES, NUM_CLASSES

try:
    _popcount = np.bitwise_count  # numpy >= 2.0
except AttributeError:  # pragma: no cover
    _TABLE = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)

    def _popcount(x):
        return _TABLE[x.view(np.uint8)].reshape(x.shape + (8,)).sum(-1)


def _bits_to_i64(bits):
    """64 booleans -> one int64 (same bit pattern as the uint64 hash)."""
    packed = np.packbits(np.asarray(bits, dtype=np.uint8).reshape(-1)[:64])
    return int(packed.view(">u8").astype(np.uint64).view(np.int64)[0])


def image_signature(img):
    """128-bit signature (dHash + aHash) of the bag region of a scan, as two int64 values.

    The hash is computed on the crop around non-white pixels, so large white margins
    (conveyor background) do not make different bags look alike."""
    import cv2
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ys, xs = np.nonzero(gray < 235)
    if len(ys) > 50:
        gray = gray[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    d = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA).astype(np.int16)
    dh = _bits_to_i64((d[:, 1:] > d[:, :-1]).flatten())
    a = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA).astype(np.float32)
    ah = _bits_to_i64((a > a.mean()).flatten())
    return dh, ah


def _as_u64(sig):
    return np.asarray(sig, dtype=np.int64).reshape(-1, 2).view(np.uint64)


def _hamming(a, b):
    """a: (n,2) u64, b: (m,2) u64 -> (n,m) summed Hamming distance over both hashes."""
    return (_popcount(a[:, None, 0] ^ b[None, :, 0]).astype(np.int16)
            + _popcount(a[:, None, 1] ^ b[None, :, 1]).astype(np.int16))


def near_duplicate_pairs(sigs, max_dist, chunk=512):
    h = _as_u64(sigs)
    n = len(h)
    out_i, out_j, out_d = [], [], []
    for s in range(0, n, chunk):
        d = _hamming(h[s:s + chunk], h)
        ii, jj = np.nonzero(d <= max_dist)
        keep = jj > ii + s
        out_i.append(ii[keep] + s)
        out_j.append(jj[keep])
        out_d.append(d[ii[keep], jj[keep]])
    if not out_i:
        return np.zeros(0, int), np.zeros(0, int), np.zeros(0, int)
    return np.concatenate(out_i), np.concatenate(out_j), np.concatenate(out_d)


def _components(n, ii, jj):
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    if len(ii) == 0:
        return np.arange(n)
    g = coo_matrix((np.ones(len(ii), np.int8), (ii, jj)), shape=(n, n))
    _, lab = connected_components(g, directed=False)
    return lab


def group_near_duplicates(sigs, max_dist=12, max_group_frac=0.02, min_dist=0, verbose=True):
    """Connected components over scan pairs whose 128-bit signatures differ by <= max_dist bits.
    Tightens the threshold if one group swallows more than max_group_frac of the scans
    (similar-looking but different bags)."""
    n = len(sigs)
    if n == 0:
        return np.zeros(0, int), max_dist
    ii, jj, dd = near_duplicate_pairs(sigs, max_dist)
    groups, thr = np.arange(n), min_dist
    for thr in range(max_dist, min_dist - 1, -1):
        sel = dd <= thr
        groups = _components(n, ii[sel], jj[sel])
        counts = np.bincount(groups)
        if counts.max() <= max(10, max_group_frac * n) or thr == min_dist:
            break
    counts = np.bincount(groups)
    if verbose:
        print(f"near-duplicate grouping: threshold={thr} bits (of 128), {len(counts)} groups, "
              f"{int((counts > 1).sum())} groups with >1 scan, largest group={counts.max()} scans")
    return groups, thr


def grouped_multilabel_split(Y, groups, val_frac=0.15, seed=0, extra_strata=None):
    """Iterative stratification (Sechidis et al., 2011) applied to groups instead of images.

    Y: (N, C) binary labels. groups: (N,) group ids. extra_strata: optional (N, K) binary columns
    (e.g. a 'benign' indicator) that should also be balanced. Returns boolean val mask (N,).
    """
    rng = np.random.default_rng(seed)
    Y = np.asarray(Y, dtype=np.float32)
    if extra_strata is not None:
        Y = np.concatenate([Y, np.asarray(extra_strata, np.float32)], 1)
    uniq, inv = np.unique(groups, return_inverse=True)
    G = len(uniq)
    GY = np.zeros((G, Y.shape[1]), np.float32)
    np.add.at(GY, inv, Y)
    gsize = np.bincount(inv, minlength=G).astype(np.float32)
    desired = np.stack([Y.sum(0) * (1 - val_frac), Y.sum(0) * val_frac])      # (2, C)
    desired_n = np.array([len(Y) * (1 - val_frac), len(Y) * val_frac])
    assign = -np.ones(G, int)
    remaining = np.ones(G, bool)
    while remaining.any():
        counts = (GY[remaining] > 0).sum(0)
        counts = np.where(counts > 0, counts, np.inf)
        if np.isinf(counts).all():
            break
        c = int(np.argmin(counts))
        cand = np.where(remaining & (GY[:, c] > 0))[0]
        rng.shuffle(cand)
        for g in cand:
            need = desired[:, c]
            best = np.flatnonzero(need == need.max())
            if len(best) > 1:
                best = best[np.flatnonzero(desired_n[best] == desired_n[best].max())]
            s = int(rng.choice(best))
            assign[g] = s
            remaining[g] = False
            desired[s] -= GY[g]
            desired_n[s] -= gsize[g]
    for g in np.where(remaining)[0]:  # label-less groups
        s = int(np.argmax(desired_n))
        assign[g] = s
        desired_n[s] -= gsize[g]
    return assign[inv] == 1


def leakage_audit(train_sigs, test_sigs, max_dist, chunk=512):
    """For each test scan, the minimum signature distance to any training scan."""
    tr, te = _as_u64(train_sigs), _as_u64(test_sigs)
    best = np.full(len(te), 128, np.int16)
    for s in range(0, len(te), chunk):
        best[s:s + chunk] = _hamming(te[s:s + chunk], tr).min(1)
    return best, best <= max_dist


def split_report(df, Y, masks: dict):
    rows = []
    for name, m in masks.items():
        y = Y[m]
        row = dict(split=name, scans=int(m.sum()), threat=int((y.sum(1) > 0).sum()),
                   clean=int((y.sum(1) == 0).sum()), multi=int((y.sum(1) > 1).sum()))
        for c, cname in enumerate(CLASS_NAMES):
            row[cname] = int(y[:, c].sum())
        rows.append(row)
    return pd.DataFrame(rows).set_index("split")


def class_ids(names):
    return [CLASS_NAMES.index(n) for n in names]


def exclude_mask(labels_series, excluded_ids):
    ex = set(int(c) for c in excluded_ids)
    return np.array([bool(ex & set(int(c) for c in labs)) for labs in labels_series], dtype=bool)


def train_subset(n, fraction, seed=0, Y=None):
    """Deterministic, label-stratified subset (same for every ablation)."""
    if fraction >= 1.0:
        return np.arange(n)
    rng = np.random.default_rng(seed + 12345)
    if Y is None:
        idx = rng.permutation(n)[: int(round(n * fraction))]
        return np.sort(idx)
    groups = np.arange(n)
    keep = grouped_multilabel_split(Y, groups, val_frac=fraction, seed=seed + 12345,
                                    extra_strata=(Y.sum(1, keepdims=True) == 0).astype(np.float32))
    return np.flatnonzero(keep)


__all__ = ["image_signature", "group_near_duplicates", "grouped_multilabel_split", "leakage_audit",
           "split_report", "class_ids", "exclude_mask", "train_subset", "NUM_CLASSES"]
