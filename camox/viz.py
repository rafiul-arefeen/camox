"""Plotting helpers (matplotlib). One palette, thin marks, hairline solid grids, no dual axes."""
from __future__ import annotations

import textwrap
from pathlib import Path

import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from .config import CLASS_NAMES
from .material import CHANNELS, material_maps_np

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
POS, NEG, MID = "#2a78d6", "#e34948", "#f0efec"   # diverging poles (blue = better, red = worse)
BLUES = LinearSegmentedColormap.from_list(
    "seq_blue", ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
GT_COLOR, PRED_COLOR = "#1baf7a", "#e87ba4"


def set_style():
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.titlesize": 11, "axes.titleweight": "semibold", "axes.labelsize": 9,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
        "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.frameon": False, "legend.fontsize": 8, "font.family": "sans-serif",
        "axes.prop_cycle": mpl.cycler(color=SERIES), "lines.linewidth": 2.0,
        "lines.solid_capstyle": "round", "figure.dpi": 110,
    })


def _save(fig, path):
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight", dpi=150)


def _noaxis(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)


# ----------------------------------------------------------------------------- EDA
def plot_class_counts(counts_by_split: dict, title="Scans per threat class", path=None):
    """counts_by_split: {'train': array(20), 'test': array(20)}"""
    names = list(counts_by_split)
    y = np.arange(len(CLASS_NAMES))
    h = 0.8 / len(names)
    fig, ax = plt.subplots(figsize=(8, 6.5))
    for k, n in enumerate(names):
        ax.barh(y + (k - (len(names) - 1) / 2) * h, counts_by_split[n], height=h * 0.9, color=SERIES[k], label=n)
    ax.set_yticks(y, CLASS_NAMES)
    ax.invert_yaxis()
    ax.set_xlabel("scans containing the class")
    ax.grid(axis="y", visible=False)
    ax.set_title(title, loc="left")
    ax.legend(loc="lower right")
    _save(fig, path)
    return fig


def plot_hist(values_by_name: dict, bins=40, log=False, xlabel="", title="", path=None):
    fig, ax = plt.subplots(figsize=(7, 3.2))
    for k, (n, v) in enumerate(values_by_name.items()):
        v = np.asarray(v, np.float64)
        v = v[np.isfinite(v)]
        if log:
            v = v[v > 0]
            b = np.logspace(np.log10(max(v.min(), 1e-6)), np.log10(v.max() + 1e-9), bins) if len(v) else bins
        else:
            b = bins
        ax.hist(v, bins=b, histtype="step", linewidth=2, color=SERIES[k], label=n)
    if log:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    ax.set_title(title, loc="left")
    if len(values_by_name) > 1:
        ax.legend()
    _save(fig, path)
    return fig


def plot_hue_hist(centers, hist, hues: dict, path=None):
    fig, ax = plt.subplots(figsize=(7, 2.8))
    ax.bar(centers, hist, width=(centers[1] - centers[0]) * 0.8, color=SERIES[0])
    for k, (name, h) in enumerate(hues.items()):
        ax.axvline(h, color=MUTED, linewidth=1)
        ax.text(h, ax.get_ylim()[1] * 0.95, f" {name}", color=INK2, fontsize=8, va="top")
    ax.set_xlim(0, 360)
    ax.set_xlabel("hue (degrees) of saturated pixels")
    ax.set_ylabel("share of pixels")
    ax.set_title("Scanner colour palette vs. the material-prior hue centres", loc="left")
    _save(fig, path)
    return fig


def draw_boxes(ax, boxes, color=GT_COLOR, names=True, lw=1.5):
    for b in boxes:
        x1, y1, x2, y2 = b[:4]
        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=lw))
        if names and len(b) > 4 and int(b[4]) >= 0:
            label = CLASS_NAMES[int(b[4])] + (f" {b[5]:.2f}" if len(b) > 5 else "")
            ax.text(x1, y1 - 2, label, color=INK, fontsize=6.5, va="bottom",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1))


def show_samples(rows, loader_fn, ncols=4, path=None, title="Sample scans (green = ground-truth boxes/mask)"):
    n = len(rows)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.5 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, r in zip(axes, rows):
        img, m = loader_fn(r["uid"])
        ax.imshow(img)
        if m is not None and m.max() > 0:
            ax.contour(m > 127, levels=[0.5], colors=[GT_COLOR], linewidths=1)
        draw_boxes(ax, r.get("boxes_cache") or [])
        labs = ", ".join(CLASS_NAMES[c] for c in r["labels"]) or "clean bag"
        ax.set_title(textwrap.shorten(f"{r['folder_name']} | {labs}", 48), fontsize=8, loc="left")
        _noaxis(ax)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, fontweight="semibold")
    fig.tight_layout()
    _save(fig, path)
    return fig


def show_material(img, path=None):
    mats = material_maps_np(img)
    fig, axes = plt.subplots(1, 7, figsize=(17, 2.8))
    axes[0].imshow(img)
    axes[0].set_title("scan", fontsize=9, loc="left")
    for i in range(6):
        axes[i + 1].imshow(mats[..., i], cmap=BLUES, vmin=0, vmax=1)
        axes[i + 1].set_title(CHANNELS[i], fontsize=9, loc="left")
    for ax in axes:
        _noaxis(ax)
    fig.suptitle("Material prior channels (darker = stronger)", x=0.01, ha="left", fontsize=11,
                 fontweight="semibold")
    fig.tight_layout()
    _save(fig, path)
    return fig


def show_augmented(ds, idxs, path=None, title="Training samples after CA-TIP + augmentation"):
    n = len(idxs)
    fig, axes = plt.subplots(2, n, figsize=(2.8 * n, 6))
    axes = np.asarray(axes).reshape(2, n)
    for j, i in enumerate(idxs):
        x, m, y, _ = ds[i]
        img = x.permute(1, 2, 0).numpy()
        axes[0, j].imshow(img)
        labs = [CLASS_NAMES[c] for c in np.flatnonzero(y.numpy())]
        axes[0, j].set_title(textwrap.shorten(", ".join(labs) or "clean", 30), fontsize=8, loc="left")
        axes[1, j].imshow(m.numpy(), cmap=BLUES, vmin=0, vmax=255)
        axes[1, j].set_title("mask target", fontsize=8, loc="left")
        _noaxis(axes[0, j])
        _noaxis(axes[1, j])
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, fontweight="semibold")
    fig.tight_layout()
    _save(fig, path)
    return fig


# ----------------------------------------------------------------------------- training & results
def plot_histories(histories: dict, key="val_mAP", path=None, title="Validation mAP per epoch"):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    for k, (name, hist) in enumerate(histories.items()):
        ep = [h["epoch"] for h in hist if key in h]
        v = [h[key] for h in hist if key in h]
        if not v:
            continue
        color = SERIES[k % len(SERIES)] if k < len(SERIES) else MUTED
        ax.plot(ep, v, color=color, label=name)
        ax.plot(ep[-1:], v[-1:], "o", color=color, markersize=5, markeredgecolor=SURFACE, markeredgewidth=2)
    ax.set_xlabel("epoch")
    ax.set_ylabel(key)
    ax.set_title(title, loc="left")
    ax.legend(ncol=2)
    _save(fig, path)
    return fig


def plot_deltas(df, ref_name, value="mAP", lo="d_lo", hi="d_hi", delta="delta", path=None,
                title="Ablations: change in test mAP vs. full model"):
    d = df.sort_values(delta)
    fig, ax = plt.subplots(figsize=(7.5, 0.38 * len(d) + 1.2))
    y = np.arange(len(d))
    colors = [NEG if v < 0 else POS for v in d[delta]]
    ax.barh(y, d[delta], height=0.55, color=colors)
    if lo in d and hi in d:
        ax.errorbar(d[delta], y, xerr=[d[delta] - d[lo], d[hi] - d[delta]], fmt="none", ecolor=INK2,
                    elinewidth=1, capsize=0)
    ax.axvline(0, color=AXIS, linewidth=1)
    ax.set_yticks(y, d["label"] if "label" in d else d.index)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(f"\u0394 {value} vs. {ref_name} (95% paired bootstrap CI)")
    ax.set_title(title, loc="left")
    _save(fig, path)
    return fig


def plot_per_class(ap_by_run: dict, path=None, title="Per-class AP on the test set"):
    names = list(ap_by_run)[:3]
    y = np.arange(len(CLASS_NAMES))
    h = 0.8 / len(names)
    fig, ax = plt.subplots(figsize=(8, 7))
    for k, n in enumerate(names):
        ax.barh(y + (k - (len(names) - 1) / 2) * h, ap_by_run[n], height=h * 0.9, color=SERIES[k], label=n)
    ax.set_yticks(y, CLASS_NAMES)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("average precision")
    ax.grid(axis="y", visible=False)
    ax.set_title(title, loc="left")
    ax.legend(loc="lower right")
    _save(fig, path)
    return fig


def plot_confusion(cm, path=None, title="Confusion on single-threat test scans (row-normalised)"):
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    im = ax.imshow(cmn, cmap=BLUES, vmin=0, vmax=1)
    ax.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=90)
    ax.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    ax.grid(False)
    ax.set_xlabel("predicted (argmax)")
    ax.set_ylabel("true class")
    for i in range(len(cmn)):
        for j in range(len(cmn)):
            if cmn[i, j] >= 0.1:
                ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if cmn[i, j] > 0.5 else INK)
    fig.colorbar(im, ax=ax, fraction=0.04)
    ax.set_title(title, loc="left")
    _save(fig, path)
    return fig


def plot_recall_clearance(is_threat, scores: dict, path=None,
                          title="Threat recall vs. share of clean bags cleared"):
    is_threat = np.asarray(is_threat, bool)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for k, (name, s) in enumerate(scores.items()):
        s = np.asarray(s, np.float64)
        pos, neg = s[is_threat], s[~is_threat]
        thr = np.unique(np.quantile(pos, np.linspace(0, 1, 400)))
        rec = [(pos >= t).mean() for t in thr]
        clear = [(neg < t).mean() for t in thr]
        ax.plot(clear, rec, color=SERIES[k % len(SERIES)], label=name)
    ax.set_xlabel("clean bags cleared (no alarm)")
    ax.set_ylabel("threat bags flagged (recall)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0.5, 1.001)
    ax.set_title(title, loc="left")
    ax.legend(loc="lower left")
    _save(fig, path)
    return fig


def plot_roc(curves: dict, path=None, title="ROC"):
    from sklearn.metrics import roc_auc_score, roc_curve
    fig, ax = plt.subplots(figsize=(5, 4.5))
    for k, (name, (t, s)) in enumerate(curves.items()):
        if len(np.unique(t)) < 2:
            continue
        fpr, tpr, _ = roc_curve(t, s)
        ax.plot(fpr, tpr, color=SERIES[k % len(SERIES)], label=f"{name} (AUROC {roc_auc_score(t, s):.3f})")
    ax.plot([0, 1], [0, 1], color=AXIS, linewidth=1)
    ax.set_xlabel("false-alarm rate on clean bags")
    ax.set_ylabel("detection rate")
    ax.set_title(title, loc="left")
    ax.legend(loc="lower right")
    _save(fig, path)
    return fig


def plot_strata(tables: dict, metric="recall", path=None, title="Recall by difficulty stratum"):
    """tables: {run: {'size': df, 'clutter': df}} with stratum/metric columns."""
    runs = list(tables)[:3]
    keys = list(tables[runs[0]].keys())
    fig, axes = plt.subplots(1, len(keys), figsize=(4.2 * len(keys), 3.4), squeeze=False)
    for a, key in zip(axes[0], keys):
        strata = list(tables[runs[0]][key]["stratum"])
        x = np.arange(len(strata))
        w = 0.8 / len(runs)
        for k, r in enumerate(runs):
            a.bar(x + (k - (len(runs) - 1) / 2) * w, tables[r][key][metric], width=w * 0.9, color=SERIES[k],
                  label=r)
        a.set_xticks(x, strata)
        a.set_ylim(0, 1)
        a.grid(axis="x", visible=False)
        a.set_title(f"{metric} by {key}", loc="left", fontsize=10)
    axes[0][0].legend(loc="lower left")
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, fontweight="semibold")
    fig.tight_layout()
    _save(fig, path)
    return fig


# ----------------------------------------------------------------------------- qualitative panels
def overlay_map(ax, img, amap, alpha=0.85, vmin=None, vmax=None):
    """Heat overlay whose opacity follows the value, so low values leave the scan visible."""
    ax.imshow(img)
    a = cv2.resize(np.asarray(amap, np.float32), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
    lo = np.nanmin(a) if vmin is None else vmin
    hi = np.nanmax(a) if vmax is None else vmax
    t = np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)
    rgba = BLUES(0.35 + 0.65 * t)
    rgba[..., 3] = alpha * t
    ax.imshow(rgba)


def qualitative_panel(img, gt_mask, pred_mask, cams, cam_names, anomaly, props_cache, crops, crop_titles,
                      title="", path=None):
    """One row: scan+GT | predicted mask | top class maps | anomaly map | proposals | zoom crops."""
    ncam = len(cams)
    ncrop = len(crops)
    ncols = 3 + ncam + (1 if anomaly is not None else 0) + max(ncrop, 1)
    fig, axes = plt.subplots(1, ncols, figsize=(2.6 * ncols, 3.0))
    k = 0
    axes[k].imshow(img)
    if gt_mask is not None and gt_mask.max() > 0:
        m = cv2.resize(gt_mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        axes[k].contour(m > 127, levels=[0.5], colors=[GT_COLOR], linewidths=1.2)
    axes[k].set_title("scan + true mask", fontsize=8, loc="left")
    k += 1
    overlay_map(axes[k], img, pred_mask.astype(np.float32) / 255.0, vmin=0, vmax=1)
    axes[k].set_title("predicted mask", fontsize=8, loc="left")
    k += 1
    for cam, name in zip(cams, cam_names):
        overlay_map(axes[k], img, cam)
        axes[k].set_title(textwrap.shorten(f"map: {name}", 26), fontsize=8, loc="left")
        k += 1
    if anomaly is not None:
        overlay_map(axes[k], img, anomaly)
        axes[k].set_title("anomaly map", fontsize=8, loc="left")
        k += 1
    axes[k].imshow(img)
    draw_boxes(axes[k], props_cache, color=PRED_COLOR, names=False)
    axes[k].set_title("Look: proposals", fontsize=8, loc="left")
    k += 1
    for c, t in zip(crops, crop_titles):
        axes[k].imshow(c)
        axes[k].set_title(textwrap.shorten(t, 26), fontsize=8, loc="left")
        k += 1
    for ax in axes:
        _noaxis(ax)
    for ax in axes[k:]:
        ax.axis("off")
    fig.suptitle(title, x=0.01, ha="left", fontsize=10, fontweight="semibold")
    fig.tight_layout()
    _save(fig, path)
    return fig


def gallery(images, titles, ncols=6, title="", path=None, boxes=None):
    n = len(images)
    if n == 0:
        return None
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.6 * ncols, 2.9 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for i, (ax, im, t) in enumerate(zip(axes, images, titles)):
        ax.imshow(im)
        if boxes is not None and boxes[i]:
            draw_boxes(ax, boxes[i], names=False)
        ax.set_title(textwrap.shorten(t, 34), fontsize=7.5, loc="left")
        _noaxis(ax)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, fontweight="semibold")
    fig.tight_layout()
    _save(fig, path)
    return fig


set_style()

__all__ = [n for n in dir() if not n.startswith("_")]
