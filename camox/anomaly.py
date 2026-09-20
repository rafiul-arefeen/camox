"""Open-set branch: PatchCore-style nearest-neighbour anomaly maps from a bank of threat-free patches.

STCray has few clean bags, so the default 'clutter' bank takes patches from ALL training scans but
skips every cell that touches (a dilated) annotated threat mask.
"""
from __future__ import annotations

import math

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchFeatures(nn.Module):
    """Frozen CNN -> locally aggregated, randomly projected patch features on the stride-16 grid."""

    def __init__(self, backbone="wide_resnet50_2.tv_in1k", layers="l23", proj_dim=384, pretrained=True, seed=0):
        super().__init__()
        self.net = timm.create_model(backbone, pretrained=pretrained, features_only=True, out_indices=(2, 3))
        self.net.eval()
        for p in self.net.parameters():
            p.requires_grad_(False)
        chs = self.net.feature_info.channels()
        self.layers = layers
        in_dim = sum(chs) if layers == "l23" else chs[1]
        g = torch.Generator().manual_seed(seed)
        self.register_buffer("proj", torch.randn(in_dim, proj_dim, generator=g) / math.sqrt(proj_dim))
        cfg = getattr(self.net, "pretrained_cfg", None) or {}
        self.register_buffer("mean", torch.tensor(cfg.get("mean", (0.485, 0.456, 0.406))).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(cfg.get("std", (0.229, 0.224, 0.225))).view(1, 3, 1, 1))
        self.pool = nn.AvgPool2d(3, 1, 1)

    def train(self, mode=True):  # always frozen
        return super().train(False)

    @torch.no_grad()
    def forward(self, x):
        x = x.float() / 255.0 if x.dtype == torch.uint8 else x.float()
        with torch.autocast(device_type=x.device.type, dtype=torch.float16, enabled=x.is_cuda):
            f2, f3 = self.net((x - self.mean) / self.std)
        f3 = self.pool(f3.float())
        if self.layers == "l23":
            f2 = F.adaptive_avg_pool2d(self.pool(f2.float()), f3.shape[-2:])
            f = torch.cat([f2, f3], 1)
        else:
            f = f3
        return torch.einsum("bchw,cd->bhwd", f, self.proj)


def _cell_masks(m, h, w, dilate):
    mm = F.adaptive_max_pool2d(m.float()[:, None] / 255.0, (h, w))[:, 0] > 0.02
    if dilate > 0:
        mm = F.max_pool2d(mm.float()[:, None], 2 * dilate + 1, 1, dilate)[:, 0] > 0
    return mm


@torch.no_grad()
def collect_bank(extractor, loader, device, mode="clutter", per_image=16, clean_per_image=64, dilate=2,
                 max_white_frac=0.1, max_total=400_000, seed=0, progress=True):
    """mode: 'clutter' (threat-free cells of every scan), 'clean' (clean bags only) or 'both'."""
    extractor = extractor.to(device).eval()
    gen = torch.Generator().manual_seed(seed)
    out, total = [], 0
    it = loader
    if progress:
        try:
            from tqdm.auto import tqdm
            it = tqdm(loader, desc=f"bank ({mode})", leave=False)
        except Exception:
            pass
    for x, m, y, _ in it:
        f = extractor(x.to(device, non_blocking=True)).float()
        B, h, w, D = f.shape
        threat_cells = _cell_masks(m.to(device), h, w, dilate)
        white = F.adaptive_avg_pool2d(x.to(device).float() / 255.0, (h, w)).min(1)[0] > 0.92
        clean = (y.sum(1) == 0).to(device)
        for b in range(B):
            if mode == "clean" and not bool(clean[b]):
                continue
            if bool(clean[b]):
                valid = torch.ones(h * w, dtype=torch.bool, device=device)
                k = clean_per_image if mode != "clutter" else per_image
            else:
                if mode == "clean":
                    continue
                valid = ~threat_cells[b].flatten()
                k = per_image
            wflat = white[b].flatten()
            idx_obj = torch.nonzero(valid & ~wflat).flatten().cpu()
            idx_wht = torch.nonzero(valid & wflat).flatten().cpu()
            k_w = min(len(idx_wht), int(round(k * max_white_frac)))
            k_o = min(len(idx_obj), k - k_w)
            pick = []
            if k_o > 0:
                pick.append(idx_obj[torch.randperm(len(idx_obj), generator=gen)[:k_o]])
            if k_w > 0:
                pick.append(idx_wht[torch.randperm(len(idx_wht), generator=gen)[:k_w]])
            if not pick:
                continue
            sel = torch.cat(pick).to(device)
            out.append(f[b].reshape(-1, D)[sel].half().cpu())
            total += len(sel)
        if total >= max_total:
            break
    if not out:
        raise RuntimeError(f"No patches collected for bank mode '{mode}'")
    return torch.cat(out)


@torch.no_grad()
def greedy_coreset(feats, k, device, proj_dim=128, seed=0):
    """Approximate greedy k-centre selection (as in PatchCore) on a random 128-d projection."""
    n = len(feats)
    if k >= n:
        return feats
    g = torch.Generator().manual_seed(seed)
    P = torch.randn(feats.shape[1], proj_dim, generator=g)
    z = torch.cat([(chunk.float() @ P) for chunk in feats.split(65536)]).to(device)
    sel = torch.empty(k, dtype=torch.long, device=device)
    sel[0] = int(torch.randint(n, (1,), generator=g))
    mind = ((z - z[sel[0]]) ** 2).sum(1)
    for i in range(1, k):
        j = torch.argmax(mind)
        sel[i] = j
        mind = torch.minimum(mind, ((z - z[j]) ** 2).sum(1))
    return feats[sel.cpu()]


def random_subsample(feats, k, seed=0):
    if k >= len(feats):
        return feats
    g = torch.Generator().manual_seed(seed)
    return feats[torch.randperm(len(feats), generator=g)[:k]]


def _gauss_kernel(sigma, device):
    r = max(1, int(round(3 * sigma)))
    t = torch.arange(-r, r + 1, device=device, dtype=torch.float32)
    k = torch.exp(-0.5 * (t / sigma) ** 2)
    return k / k.sum(), r


@torch.no_grad()
def score_loader(extractor, bank, loader, device, topk=10, blur_sigma=1.0, progress=True):
    """Returns per-image max and top-k-mean NN distance plus the (h,w) distance maps (float16)."""
    extractor = extractor.to(device).eval()
    B_ = bank.to(device).float()
    B_sq = (B_ ** 2).sum(1)
    k1, r = _gauss_kernel(blur_sigma, device) if blur_sigma > 0 else (None, 0)
    smax, stop, maps = [], [], []
    it = loader
    if progress:
        try:
            from tqdm.auto import tqdm
            it = tqdm(loader, desc="anomaly scores", leave=False)
        except Exception:
            pass
    for x, _, _, _ in it:
        f = extractor(x.to(device, non_blocking=True)).float()
        Bn, h, w, D = f.shape
        q = f.reshape(Bn, -1, D)
        dm = torch.empty(Bn, h * w, device=device)
        for b in range(Bn):
            qq = q[b]
            d2 = (qq ** 2).sum(1, keepdim=True) + B_sq[None] - 2.0 * (qq @ B_.T)
            dm[b] = d2.clamp_min_(0).min(1)[0].sqrt_()
        dmap = dm.reshape(Bn, 1, h, w)
        if k1 is not None:
            dmap = F.conv2d(F.pad(dmap, (r, r, 0, 0), mode="replicate"), k1.view(1, 1, 1, -1))
            dmap = F.conv2d(F.pad(dmap, (0, 0, r, r), mode="replicate"), k1.view(1, 1, -1, 1))
        flat = dmap.flatten(1)
        smax.append(flat.max(1)[0].cpu())
        stop.append(flat.topk(min(topk, flat.shape[1]), 1)[0].mean(1).cpu())
        maps.append(dmap[:, 0].half().cpu())
    return dict(score_max=torch.cat(smax).numpy(), score_topk=torch.cat(stop).numpy(),
                maps=torch.cat(maps).numpy())


def robust_fit(x):
    x = np.asarray(x, np.float64)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return (0.0, 1.0)
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med))) * 1.4826
    if mad < 1e-6:
        mad = float(x.std()) + 1e-6
    return (med, mad)


def zscore(x, fit):
    return (np.asarray(x, np.float64) - fit[0]) / fit[1]


def masked_max(maps, masks, dilate=2):
    """Max anomaly over threat-free cells of threat scans: a large 'clean-like' null distribution."""
    m = torch.from_numpy(np.ascontiguousarray(masks))
    h, w = maps.shape[1:]
    cells = _cell_masks(m, h, w, dilate).numpy()
    out = np.full(len(maps), np.nan, np.float32)
    for i in range(len(maps)):
        v = maps[i][~cells[i]]
        if v.size:
            out[i] = v.max()
    return out


def clutter_cell_stats(maps, masks, dilate=2, max_n=500_000, seed=0):
    m = torch.from_numpy(np.ascontiguousarray(masks))
    h, w = maps.shape[1:]
    cells = _cell_masks(m, h, w, dilate).numpy()
    vals = maps[~cells].astype(np.float32)
    if len(vals) > max_n:
        vals = np.random.default_rng(seed).choice(vals, max_n, replace=False)
    return robust_fit(vals)


def upsample_maps(maps, size):
    t = torch.from_numpy(maps.astype(np.float32))[:, None]
    return F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)[:, 0].numpy()


__all__ = ["PatchFeatures", "collect_bank", "greedy_coreset", "random_subsample", "score_loader", "robust_fit",
           "zscore", "masked_max", "clutter_cell_stats", "upsample_maps"]
