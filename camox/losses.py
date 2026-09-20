"""Loss functions: ASL, focal, class-balanced BCE, BCE, and BCE + soft Dice for the threat mask."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class AsymmetricLoss(nn.Module):
    """Ridnik et al., ICCV 2021. Sum over classes, mean over the batch."""

    def __init__(self, gamma_neg=4.0, gamma_pos=0.0, clip=0.05, eps=1e-8):
        super().__init__()
        self.gamma_neg, self.gamma_pos, self.clip, self.eps = gamma_neg, gamma_pos, clip, eps

    def forward(self, logits, y):
        x = logits.float()
        y = y.float()
        p = torch.sigmoid(x)
        p_neg = 1.0 - p
        if self.clip and self.clip > 0:
            p_neg = (p_neg + self.clip).clamp(max=1.0)
        loss = y * torch.log(p.clamp(min=self.eps)) + (1 - y) * torch.log(p_neg.clamp(min=self.eps))
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            with torch.no_grad():
                pt = p * y + p_neg * (1 - y)
                w = torch.pow(1 - pt, self.gamma_pos * y + self.gamma_neg * (1 - y))
            loss = loss * w
        return -loss.sum() / x.shape[0]


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma, self.alpha = gamma, alpha

    def forward(self, logits, y):
        x, y = logits.float(), y.float()
        ce = F.binary_cross_entropy_with_logits(x, y, reduction="none")
        p = torch.sigmoid(x)
        pt = p * y + (1 - p) * (1 - y)
        loss = ce * (1 - pt) ** self.gamma
        if self.alpha is not None:
            loss = loss * (self.alpha * y + (1 - self.alpha) * (1 - y))
        return loss.sum() / x.shape[0]


class WeightedBCE(nn.Module):
    """BCE with optional per-class weights (class-balanced variant of Cui et al., CVPR 2019)."""

    def __init__(self, class_weights=None):
        super().__init__()
        w = torch.ones(1) if class_weights is None else torch.as_tensor(class_weights, dtype=torch.float32)
        self.register_buffer("w", w)

    def forward(self, logits, y):
        x, y = logits.float(), y.float()
        loss = F.binary_cross_entropy_with_logits(x, y, reduction="none") * self.w
        return loss.sum() / x.shape[0]


def class_balanced_weights(pos_counts, beta=0.999):
    n = np.maximum(np.asarray(pos_counts, np.float64), 1.0)
    eff = (1.0 - np.power(beta, n)) / (1.0 - beta)
    w = 1.0 / eff
    return (w / w.sum() * len(w)).astype(np.float32)


def seg_loss(logits, target, smooth=1.0):
    """BCE + global soft Dice. target: soft mask in [0,1] at the logits' resolution."""
    x = logits.float().squeeze(1)
    t = target.float()
    if t.shape[-2:] != x.shape[-2:]:
        t = F.interpolate(t[:, None], size=x.shape[-2:], mode="area")[:, 0]
    bce = F.binary_cross_entropy_with_logits(x, t)
    p = torch.sigmoid(x)
    inter = (p * t).sum()
    dice = 1.0 - (2 * inter + smooth) / (p.sum() + t.sum() + smooth)
    return bce + dice


def build_cls_loss(cfg, pos_counts=None):
    name = cfg.get("loss", "asl")
    if name == "asl":
        return AsymmetricLoss(cfg.get("asl_gamma_neg", 4.0), cfg.get("asl_gamma_pos", 0.0), cfg.get("asl_clip", 0.05))
    if name == "focal":
        return FocalLoss(cfg.get("focal_gamma", 2.0))
    if name == "cb":
        return WeightedBCE(class_balanced_weights(pos_counts, cfg.get("cb_beta", 0.999)))
    if name == "bce":
        return WeightedBCE(None)
    raise ValueError(f"unknown loss {name}")
