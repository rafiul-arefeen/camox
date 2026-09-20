"""CAMO-X networks: material-gated dual-stream backbone, FPN, CSRA class head and threat-mask head."""
from __future__ import annotations

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import NUM_CLASSES
from .material import MaterialPrior


def norm(c):
    """GroupNorm for the new layers: independent of the (small) micro-batch used on an 8 GB GPU."""
    for g in (8, 4, 2, 1):
        if c % g == 0:
            return nn.GroupNorm(g, c)
    return nn.GroupNorm(1, c)


def conv_bn_act(cin, cout, k=3, s=1):
    return nn.Sequential(nn.Conv2d(cin, cout, k, s, k // 2, bias=False), norm(cout), nn.SiLU(inplace=True))


class MaterialStream(nn.Module):
    """Light CNN over the 6 material channels, returning features at strides 8/16/32."""

    def __init__(self, cin=6, widths=(32, 64, 128, 256)):
        super().__init__()
        w0, w1, w2, w3 = widths
        self.stem = nn.Sequential(conv_bn_act(cin, w0, 3, 2), conv_bn_act(w0, w0, 3, 2))       # /4
        self.s8 = nn.Sequential(conv_bn_act(w0, w1, 3, 2), conv_bn_act(w1, w1))                # /8
        self.s16 = nn.Sequential(conv_bn_act(w1, w2, 3, 2), conv_bn_act(w2, w2))               # /16
        self.s32 = nn.Sequential(conv_bn_act(w2, w3, 3, 2), conv_bn_act(w3, w3))               # /32
        self.channels = [w1, w2, w3]

    def forward(self, m):
        x = self.stem(m)
        f8 = self.s8(x)
        f16 = self.s16(f8)
        f32 = self.s32(f16)
        return [f8, f16, f32]


class GatedFusion(nn.Module):
    """F <- F + gamma * sigmoid(conv[F, P(M)]) * P(M), gamma initialised at 0 ('gated'),
    or F <- F + P(M) ('add')."""

    def __init__(self, c_rgb, c_mat, mode="gated"):
        super().__init__()
        self.mode = mode
        self.proj = nn.Sequential(nn.Conv2d(c_mat, c_rgb, 1, bias=False), norm(c_rgb))
        if mode == "gated":
            self.gate = nn.Conv2d(2 * c_rgb, c_rgb, 1)
            nn.init.zeros_(self.gate.weight)
            nn.init.zeros_(self.gate.bias)
            self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, f, m):
        if m.shape[-2:] != f.shape[-2:]:
            m = F.interpolate(m, size=f.shape[-2:], mode="bilinear", align_corners=False)
        p = self.proj(m)
        if self.mode == "add":
            return f + p
        g = torch.sigmoid(self.gate(torch.cat([f, p], 1)))
        return f + self.gamma * g * p


class FPN(nn.Module):
    def __init__(self, in_chs, out_ch=128):
        super().__init__()
        self.lat = nn.ModuleList([nn.Conv2d(c, out_ch, 1) for c in in_chs])
        self.smooth = nn.ModuleList([conv_bn_act(out_ch, out_ch) for _ in in_chs])

    def forward(self, feats):
        lat = [l(f) for l, f in zip(self.lat, feats)]
        for i in range(len(lat) - 2, -1, -1):
            lat[i] = lat[i] + F.interpolate(lat[i + 1], size=lat[i].shape[-2:], mode="nearest")
        return [s(x) for s, x in zip(self.smooth, lat)]


class CSRAHead(nn.Module):
    """Class-specific residual attention (Zhu & Wu, ICCV 2021). mode='gap' -> plain average pooling."""

    def __init__(self, cin, num_classes=NUM_CLASSES, lam=1.0, temps=(1, 99), mode="csra"):
        super().__init__()
        self.cls = nn.Conv2d(cin, num_classes, 1)
        nn.init.normal_(self.cls.weight, std=0.01)
        nn.init.constant_(self.cls.bias, -2.0)
        self.lam, self.temps, self.mode = lam, tuple(temps), mode

    def forward(self, x):
        s = self.cls(x)
        flat = s.flatten(2)
        base = flat.mean(-1)
        if self.mode == "gap":
            return base, s
        att = 0.0
        for t in self.temps:
            if t >= 99:
                att = att + flat.max(-1)[0]
            else:
                att = att + (torch.softmax(flat * t, -1) * flat).sum(-1)
        return base + self.lam * att / len(self.temps), s


class SegHead(nn.Module):
    def __init__(self, cin, mid=64):
        super().__init__()
        self.body = conv_bn_act(cin, mid)
        self.out = nn.Conv2d(mid, 1, 1)
        nn.init.constant_(self.out.bias, -4.0)

    def forward(self, x):
        return F.interpolate(self.out(self.body(x)), scale_factor=2, mode="bilinear", align_corners=False)


def _stride_indices(backbone):
    probe = timm.create_model(backbone, pretrained=False, features_only=True)
    reds = probe.feature_info.reduction()
    idx = [reds.index(r) for r in (8, 16, 32) if r in reds]
    if len(idx) != 3:
        raise ValueError(f"{backbone} does not expose stride 8/16/32 features: {reds}")
    del probe
    return tuple(idx)


def _norm_stats(model):
    cfg = getattr(model, "pretrained_cfg", None) or getattr(model, "default_cfg", None) or {}
    mean = cfg.get("mean", (0.485, 0.456, 0.406))
    std = cfg.get("std", (0.229, 0.224, 0.225))
    return torch.tensor(mean).view(1, 3, 1, 1), torch.tensor(std).view(1, 3, 1, 1)


class CamoXNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.material = cfg.get("material", "gated")
        self.input_mode = cfg.get("input_mode", "rgb")
        in_chans = 3 + (6 if self.material == "early" else 0)
        idx = _stride_indices(cfg["backbone"])
        self.backbone = timm.create_model(cfg["backbone"], pretrained=cfg.get("pretrained", True),
                                          features_only=True, out_indices=idx, in_chans=in_chans)
        chs = self.backbone.feature_info.channels()
        mean, std = _norm_stats(self.backbone)
        self.register_buffer("mean", mean, persistent=False)
        self.register_buffer("std", std, persistent=False)
        self.prior = MaterialPrior() if self.material != "none" else None
        if self.material in ("gated", "add"):
            self.mstream = MaterialStream()
            self.fuse = nn.ModuleList([GatedFusion(c, mc, self.material)
                                       for c, mc in zip(chs, self.mstream.channels)])
        fc = cfg.get("fpn_ch", 128)
        self.fpn = FPN(chs, fc)
        self.cls_neck = conv_bn_act(2 * fc, 256)
        self.head = CSRAHead(256, NUM_CLASSES, cfg.get("csra_lambda", 1.0), cfg.get("csra_temps", (1, 99)),
                             cfg.get("head", "csra"))
        self.seg_head = SegHead(fc) if cfg.get("seg", True) else None

    def new_param_groups(self):
        bb = list(self.backbone.parameters())
        bb_ids = {id(p) for p in bb}
        other = [p for p in self.parameters() if id(p) not in bb_ids]
        return bb, other

    def prep(self, x):
        x = x.float() / 255.0 if x.dtype == torch.uint8 else x.float()
        if self.input_mode == "gray":
            g = 0.299 * x[:, :1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
            x = g.expand(-1, 3, -1, -1)
        return x

    def forward(self, x):
        x = self.prep(x)
        mat = self.prior(x) if self.prior is not None else None
        xn = (x - self.mean) / self.std
        if self.material == "early":
            xn = torch.cat([xn, (mat - 0.5) / 0.25], 1)
        feats = self.backbone(xn)
        if self.material in ("gated", "add"):
            mfeats = self.mstream(mat.to(xn.dtype))
            feats = [fu(f, m) for fu, f, m in zip(self.fuse, feats, mfeats)]
        p8, p16, p32 = self.fpn(feats)
        q = torch.cat([p16, F.interpolate(p32, size=p16.shape[-2:], mode="nearest")], 1)
        logits, cams = self.head(self.cls_neck(q))
        out = {"logits": logits, "cams": cams}
        if self.seg_head is not None:
            out["seg"] = self.seg_head(p8)
        return out

    def gate_values(self):
        if self.material != "gated":
            return {}
        return {f"gamma_s{2 ** (i + 3)}": float(fu.gamma.detach().cpu()) for i, fu in enumerate(self.fuse)}


class SimpleClassifier(nn.Module):
    """Plain timm classifier (global pooling + linear): the standard fine-tuning baseline."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.input_mode = cfg.get("input_mode", "rgb")
        kw = {}
        if "vit" in cfg["backbone"] or "deit" in cfg["backbone"]:
            kw["img_size"] = cfg["img_size"]
        self.net = timm.create_model(cfg["backbone"], pretrained=cfg.get("pretrained", True),
                                     num_classes=NUM_CLASSES, **kw)
        mean, std = _norm_stats(self.net)
        self.register_buffer("mean", mean, persistent=False)
        self.register_buffer("std", std, persistent=False)

    def new_param_groups(self):
        head = self.net.get_classifier()
        head_ids = {id(p) for p in head.parameters()}
        bb = [p for p in self.parameters() if id(p) not in head_ids]
        return bb, list(head.parameters())

    def forward(self, x):
        x = x.float() / 255.0 if x.dtype == torch.uint8 else x.float()
        if self.input_mode == "gray":
            g = 0.299 * x[:, :1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
            x = g.expand(-1, 3, -1, -1)
        return {"logits": self.net((x - self.mean) / self.std)}

    def gradcam(self, x, cls_idx):
        """Grad-CAM on the last feature map (CNN backbones only)."""
        x = x.float() / 255.0 if x.dtype == torch.uint8 else x.float()
        feats = self.net.forward_features((x - self.mean) / self.std)
        if feats.ndim != 4:
            return None
        feats.retain_grad()
        logits = self.net.forward_head(feats)
        score = logits[torch.arange(len(x)), cls_idx].sum()
        self.net.zero_grad(set_to_none=True)
        score.backward()
        w = feats.grad.mean((2, 3), keepdim=True)
        cam = F.relu((w * feats).sum(1))
        return cam.detach()


def build_model(cfg):
    if cfg.get("arch", "camox") == "simple":
        return SimpleClassifier(cfg)
    return CamoXNet(cfg)


def count_params(model):
    return sum(p.numel() for p in model.parameters()) / 1e6


__all__ = ["CamoXNet", "SimpleClassifier", "build_model", "count_params", "MaterialStream", "GatedFusion"]
