"""Physics-informed material prior computed from the dual-energy pseudo-colour (no learned weights).

Most dual-energy baggage scanners colour-code effective atomic number:
    orange/yellow  -> organic (low Z: plastics, explosives, 3D-printed polymer)
    green          -> inorganic / mixed (medium Z)
    blue           -> metal (high Z)
    black          -> too dense to penetrate
The hue centres below follow that convention; check them against the hue histogram in the
EDA section and adjust MATERIAL_HUES if your scanner's palette differs.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MATERIAL_HUES = dict(organic=30.0, inorganic=110.0, metal=215.0)       # degrees
MATERIAL_WIDTHS = dict(organic=22.0, inorganic=35.0, metal=35.0)       # degrees (Gaussian sigma)
CHANNELS = ["organic", "inorganic", "metal", "density", "very_dense", "edges"]


def rgb_to_hsv_torch(x, eps=1e-6):
    """x: (B,3,H,W) in [0,1] -> h in [0,1), s, v."""
    r, g, b = x[:, 0], x[:, 1], x[:, 2]
    maxc, _ = x.max(1)
    minc, _ = x.min(1)
    v = maxc
    delta = maxc - minc
    s = delta / (maxc + eps)
    rc = (maxc - r) / (delta + eps)
    gc = (maxc - g) / (delta + eps)
    bc = (maxc - b) / (delta + eps)
    h = torch.where(maxc == r, bc - gc, torch.where(maxc == g, 2.0 + rc - bc, 4.0 + gc - rc))
    h = torch.remainder(h / 6.0, 1.0)
    return h, s, v


class MaterialPrior(nn.Module):
    """RGB (0..1) -> 6 material channels, all roughly in [0, 1]."""

    def __init__(self, hues=None, widths=None, sat_lo=0.10, sat_hi=0.35, dense_v=0.18):
        super().__init__()
        hues = hues or MATERIAL_HUES
        widths = widths or MATERIAL_WIDTHS
        self.register_buffer("centers", torch.tensor([hues[k] / 360.0 for k in ("organic", "inorganic", "metal")]))
        self.register_buffer("widths", torch.tensor([widths[k] / 360.0 for k in ("organic", "inorganic", "metal")]))
        kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]) / 4.0
        self.register_buffer("sobel", torch.stack([kx, kx.t()])[:, None])
        self.sat_lo, self.sat_hi, self.dense_v = sat_lo, sat_hi, dense_v
        self.out_channels = len(CHANNELS)

    @torch.no_grad()
    def forward(self, x):
        x = x.float()
        h, s, v = rgb_to_hsv_torch(x)
        d = torch.remainder(h[:, None] - self.centers[None, :, None, None] + 0.5, 1.0) - 0.5
        mem = torch.exp(-0.5 * (d / self.widths[None, :, None, None]) ** 2)
        gate = ((s - self.sat_lo) / (self.sat_hi - self.sat_lo)).clamp(0, 1)
        mem = mem * gate[:, None]
        density = (1.0 - v)[:, None]
        very_dense = torch.sigmoid((self.dense_v - v) * 30.0)[:, None]
        lum = (0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2])[:, None]
        g = F.conv2d(F.pad(lum, (1, 1, 1, 1), mode="replicate"), self.sobel)
        edges = (g.float().pow(2).sum(1, keepdim=True).sqrt() * 4.0).clamp(0, 1)
        return torch.cat([mem.float(), density.float(), very_dense.float(), edges], 1)


_PRIOR = None


def material_maps_np(rgb_uint8):
    """Numpy helper for visualisation: (H,W,3) uint8 -> (H,W,6) float32."""
    global _PRIOR
    if _PRIOR is None:
        _PRIOR = MaterialPrior()
    x = torch.from_numpy(rgb_uint8).permute(2, 0, 1)[None].float() / 255.0
    return _PRIOR(x)[0].permute(1, 2, 0).numpy()


def hue_histogram(images_rgb, bins=72, sat_min=0.25, val_min=0.15):
    """Histogram of hue (degrees) over saturated, non-black pixels of a list of RGB uint8 images."""
    import cv2
    hist = np.zeros(bins, np.float64)
    for im in images_rgb:
        hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV_FULL)
        h = hsv[..., 0].astype(np.float32) * 360.0 / 256.0
        s = hsv[..., 1] / 255.0
        v = hsv[..., 2] / 255.0
        sel = (s > sat_min) & (v > val_min)
        hist += np.histogram(h[sel], bins=bins, range=(0, 360))[0]
    return np.linspace(0, 360, bins, endpoint=False) + 180.0 / bins, hist / max(hist.sum(), 1)


def clutter_map(rgb_uint8, size=64):
    """Coarse clutter density (attenuation + edges) used for concealment-aware placement."""
    import cv2
    small = cv2.resize(rgb_uint8, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    v = small.max(2)
    lum = small @ np.array([0.299, 0.587, 0.114], np.float32)
    gx = cv2.Sobel(lum, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(lum, cv2.CV_32F, 0, 1, ksize=3)
    edge = np.sqrt(gx ** 2 + gy ** 2)
    dens = cv2.GaussianBlur(1.0 - v, (0, 0), 1.5)
    edge = cv2.GaussianBlur(edge, (0, 0), 1.5)
    bag = cv2.GaussianBlur((v < 0.92).astype(np.float32), (0, 0), 1.0) > 0.3
    return dens + 0.5 * edge / (edge.max() + 1e-6), bag


def local_clutter_score(rgb_uint8, box, ring=0.5):
    """Mean attenuation + edge density in a ring around a box (cache coordinates)."""
    import cv2
    h, w = rgb_uint8.shape[:2]
    x1, y1, x2, y2 = [float(t) for t in box[:4]]
    bw, bh = x2 - x1, y2 - y1
    X1, Y1 = int(max(0, x1 - ring * bw)), int(max(0, y1 - ring * bh))
    X2, Y2 = int(min(w, math.ceil(x2 + ring * bw))), int(min(h, math.ceil(y2 + ring * bh)))
    if X2 <= X1 or Y2 <= Y1:
        return float("nan"), float("nan")
    reg = rgb_uint8[Y1:Y2, X1:X2].astype(np.float32) / 255.0
    inner = np.zeros(reg.shape[:2], bool)
    inner[int(y1) - Y1:int(math.ceil(y2)) - Y1, int(x1) - X1:int(math.ceil(x2)) - X1] = True
    ring_m = ~inner
    v = reg.max(2)
    lum = reg @ np.array([0.299, 0.587, 0.114], np.float32)
    gx = cv2.Sobel(lum, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(lum, cv2.CV_32F, 0, 1, ksize=3)
    edge = np.sqrt(gx ** 2 + gy ** 2)
    if ring_m.sum() < 10:
        ring_m = np.ones_like(ring_m)
    dens_ring = float((1 - v)[ring_m].mean())
    dens_box = float((1 - v)[inner].mean()) if inner.any() else float("nan")
    return dens_ring + 0.5 * float(edge[ring_m].mean()), dens_box
