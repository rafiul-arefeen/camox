"""PyTorch datasets (kept in a module so Windows DataLoader workers can import them)."""
from __future__ import annotations

import random
from collections import defaultdict

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .cache import PATCH_CONTEXT, PATCH_SIZE, cache_paths
from .config import NUM_CLASSES
from .material import clutter_map
from .utils import box_ioa, box_iou, crop_with_pad, imread_any, imread_rgb


def _photometric(img, hue_jitter=False, rng=random):
    f = img.astype(np.float32) / 255.0
    if rng.random() < 0.8:
        gamma = rng.uniform(0.85, 1.15)
        f = np.power(np.clip(f, 0, 1), gamma)
        c = rng.uniform(0.9, 1.1)
        b = rng.uniform(-0.04, 0.04)
        f = (f - 0.5) * c + 0.5 + b
    f = np.clip(f, 0, 1)
    if hue_jitter and rng.random() < 0.8:
        hsv = cv2.cvtColor((f * 255).astype(np.uint8), cv2.COLOR_RGB2HSV_FULL).astype(np.float32)
        hsv[..., 0] = np.mod(hsv[..., 0] + rng.uniform(-24, 24), 256)
        hsv[..., 1] = np.clip(hsv[..., 1] * rng.uniform(0.7, 1.3), 0, 255)
        f = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB_FULL).astype(np.float32) / 255.0
    if rng.random() < 0.2:
        f = f + np.random.normal(0, 0.01, f.shape).astype(np.float32)
    return (np.clip(f, 0, 1) * 255.0 + 0.5).astype(np.uint8)


def _to_gray3(img):
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return np.repeat(g[..., None], 3, axis=2)


class Stage1Dataset(Dataset):
    """Letterboxed scans + soft masks + multi-label targets, with optional CA-TIP augmentation.

    items: list of (uid, labels: list[int], boxes_cache: list[[x1,y1,x2,y2,c]])
    donors: DataFrame/list of dicts with 'path' and 'cls' (only needed when tip != 'none').
    """

    def __init__(self, items, cache_dir, cache_size, img_size, train=False, tip="none", tip_prob=0.5,
                 donors=None, hue_jitter=False, geo_aug=True, input_mode="rgb", excluded=()):
        self.items = list(items)
        self.img_dir, self.mask_dir = cache_paths(cache_dir, cache_size)
        self.cache_size, self.img_size = cache_size, img_size
        self.train, self.tip, self.tip_prob = train, tip, tip_prob
        self.hue_jitter, self.geo_aug, self.input_mode = hue_jitter, geo_aug, input_mode
        self.mask_res = max(1, img_size // 4)
        excluded = set(int(c) for c in excluded)
        by = defaultdict(list)
        if donors is not None and tip != "none":
            recs = donors.to_dict("records") if hasattr(donors, "to_dict") else donors
            for d in recs:
                if int(d["cls"]) not in excluded:
                    by[int(d["cls"])].append(str(d["path"]))
        self.donors = dict(by)
        self.donor_classes = sorted(self.donors)

    def __len__(self):
        return len(self.items)

    # ------------------------------------------------------------------ helpers
    def _load(self, uid):
        img = imread_rgb(self.img_dir / f"{uid}.jpg")
        m = imread_any(self.mask_dir / f"{uid}.png", cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.full((self.cache_size, self.cache_size, 3), 255, np.uint8)
        if m is None or m.shape[:2] != img.shape[:2]:
            m = np.zeros(img.shape[:2], np.uint8)
        return img, m

    def _placement_prob(self, img):
        clut, bag = clutter_map(img, 64)
        if self.tip == "concealed":
            w = np.power(np.clip(clut, 0, None), 2.0) * bag
        else:
            w = bag.astype(np.float32)
        if w.sum() <= 1e-6:
            w = np.ones_like(w, dtype=np.float32)
        w = w.astype(np.float64).ravel()
        return w / w.sum()

    def _apply_tip(self, img, m, boxes):
        c = random.choice(self.donor_classes)
        d = imread_any(random.choice(self.donors[c]), cv2.IMREAD_UNCHANGED)
        if d is None or d.ndim != 3 or d.shape[2] != 4:
            return img, m, None
        d = cv2.cvtColor(d, cv2.COLOR_BGRA2RGBA).astype(np.float32) / 255.0
        d = np.rot90(d, random.randint(0, 3))
        if random.random() < 0.5:
            d = d[:, ::-1]
        d = np.ascontiguousarray(d)
        h, w = d.shape[:2]
        s = random.uniform(0.75, 1.25)  # TIP happens at cache resolution, before any resize
        ang = random.uniform(-15, 15)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, s)
        cos, sin = abs(M[0, 0]), abs(M[0, 1])
        nw, nh = int(h * sin + w * cos) + 1, int(h * cos + w * sin) + 1
        M[0, 2] += nw / 2 - w / 2
        M[1, 2] += nh / 2 - h / 2
        rgb = cv2.warpAffine(d[..., :3], M, (nw, nh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                             borderValue=(1.0, 1.0, 1.0))
        a = cv2.warpAffine(d[..., 3], M, (nw, nh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                           borderValue=0.0)
        H, W = img.shape[:2]
        lim = 0.45 * min(H, W)
        if max(nh, nw) > lim:
            f = lim / max(nh, nw)
            nw, nh = max(2, int(nw * f)), max(2, int(nh * f))
            rgb = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
            a = cv2.resize(a, (nw, nh), interpolation=cv2.INTER_AREA)
        if nh < 4 or nw < 4 or a.max() < 0.5:
            return img, m, None
        prob = self._placement_prob(img)
        gt = np.asarray([b[:4] for b in boxes], np.float32).reshape(-1, 4)
        x0 = y0 = 0
        for _ in range(6):
            k = int(np.random.choice(len(prob), p=prob))
            gy, gx = divmod(k, 64)
            cx = (gx + random.random()) * W / 64.0
            cy = (gy + random.random()) * H / 64.0
            x0 = int(np.clip(cx - nw / 2, 0, W - nw))
            y0 = int(np.clip(cy - nh / 2, 0, H - nh))
            if len(gt) == 0 or box_iou(np.array([[x0, y0, x0 + nw, y0 + nh]], np.float32), gt).max() < 0.2:
                break
        region = img[y0:y0 + nh, x0:x0 + nw].astype(np.float32) / 255.0
        t_eff = 1.0 - a[..., None] * (1.0 - rgb)
        img = img.copy()
        m = m.copy()
        img[y0:y0 + nh, x0:x0 + nw] = (np.clip(region * t_eff, 0, 1) * 255.0 + 0.5).astype(np.uint8)
        m[y0:y0 + nh, x0:x0 + nw] = np.maximum(m[y0:y0 + nh, x0:x0 + nw], (a * 255.0).astype(np.uint8))
        return img, m, c

    @staticmethod
    def _geo(img, m):
        if random.random() < 0.5:
            img, m = img[:, ::-1], m[:, ::-1]
        if random.random() < 0.5:
            img, m = img[::-1], m[::-1]
        k = random.randint(0, 3)
        if k:
            img, m = np.rot90(img, k), np.rot90(m, k)
        img, m = np.ascontiguousarray(img), np.ascontiguousarray(m)
        if random.random() < 0.5:  # zoom-out + shift: never crops content, so labels stay valid
            H = img.shape[0]
            s = random.uniform(0.8, 1.0)
            n = max(8, int(H * s))
            tx, ty = random.randint(0, H - n), random.randint(0, H - n)
            ci = np.full_like(img, 255)
            cm = np.zeros_like(m)
            ci[ty:ty + n, tx:tx + n] = cv2.resize(img, (n, n), interpolation=cv2.INTER_AREA)
            cm[ty:ty + n, tx:tx + n] = cv2.resize(m, (n, n), interpolation=cv2.INTER_AREA)
            img, m = ci, cm
        return img, m

    def __getitem__(self, i):
        uid, labels, boxes = self.items[i]
        img, m = self._load(uid)
        y = np.zeros(NUM_CLASSES, np.float32)
        for c in labels:
            y[int(c)] = 1.0
        if self.train:
            if self.tip != "none" and self.donor_classes and random.random() < self.tip_prob:
                img, m, c = self._apply_tip(img, m, boxes or [])
                if c is not None:
                    y[c] = 1.0
            if self.geo_aug:
                img, m = self._geo(img, m)
            img = _photometric(img, self.hue_jitter)
        if self.input_mode == "gray":
            img = _to_gray3(img)
        if img.shape[0] != self.img_size:
            img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        mt = cv2.resize(m, (self.mask_res, self.mask_res), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).contiguous()
        return x, torch.from_numpy(np.ascontiguousarray(mt)), torch.from_numpy(y), i


class CropDataset(Dataset):
    """Stage-2 zoom crops taken from stored 320x320 patches (box = centre half of the patch).

    rows: list of (patch_path, boxes_in_patch: list[[x1,y1,x2,y2,c]])
    """

    def __init__(self, rows, img_size=224, train=False, context=1.3, train_context=(1.1, 1.8), jitter=0.15,
                 input_mode="rgb", hue_jitter=False, excluded=()):
        self.rows = list(rows)
        self.img_size, self.train = img_size, train
        self.context, self.train_context, self.jitter = context, tuple(train_context), jitter
        self.input_mode, self.hue_jitter = input_mode, hue_jitter
        self.excluded = set(int(c) for c in excluded)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        path, boxes = self.rows[i]
        p = imread_rgb(path)
        if p is None:
            p = np.full((PATCH_SIZE, PATCH_SIZE, 3), 255, np.uint8)
        S = PATCH_SIZE / PATCH_CONTEXT
        if self.train:
            k = random.uniform(*self.train_context)
            dx, dy = [random.uniform(-self.jitter, self.jitter) * S for _ in range(2)]
        else:
            k, dx, dy = self.context, 0.0, 0.0
        side = k * S
        cx, cy = PATCH_SIZE / 2 + dx, PATCH_SIZE / 2 + dy
        region = (cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2)
        crop = crop_with_pad(p, region, self.img_size, 255)
        y = np.zeros(NUM_CLASSES, np.float32)
        if boxes:
            bb = np.asarray(boxes, np.float32).reshape(-1, 5)
            ioa = box_ioa(bb[:, :4], np.asarray([region], np.float32))[:, 0]
            for b, v in zip(bb, ioa):
                c = int(b[4])
                if v >= 0.5 and c >= 0 and c not in self.excluded:
                    y[c] = 1.0
        if self.train:
            if random.random() < 0.5:
                crop = crop[:, ::-1]
            if random.random() < 0.5:
                crop = crop[::-1]
            kk = random.randint(0, 3)
            if kk:
                crop = np.rot90(crop, kk)
            crop = _photometric(np.ascontiguousarray(crop), self.hue_jitter)
        if self.input_mode == "gray":
            crop = _to_gray3(np.ascontiguousarray(crop))
        x = torch.from_numpy(np.ascontiguousarray(crop)).permute(2, 0, 1).contiguous()
        return x, torch.zeros(1, 1, dtype=torch.uint8), torch.from_numpy(y), i


__all__ = ["Stage1Dataset", "CropDataset"]
