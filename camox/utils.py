"""Small helpers shared by every CAMO-X module (Windows-safe image IO, seeding, letterboxing)."""
from __future__ import annotations

import json
import math
import os
import random
import re
import time
from pathlib import Path

import cv2
import numpy as np
import torch

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


# ----------------------------------------------------------------------------- IO
def imread_any(path, flags=cv2.IMREAD_COLOR):
    """cv2.imread that also works with non-ASCII Windows paths. Returns None on failure."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, flags)
    except Exception:
        return None


def imread_rgb(path):
    img = imread_any(path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def imwrite_any(path, img, params=None):
    """cv2.imwrite for any path. `img` is BGR/gray uint8."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower() or ".png"
    ok, buf = cv2.imencode(ext, img, params or [])
    if not ok:
        raise IOError(f"Could not encode {path}")
    tmp = path.with_name(path.name + ".tmp")   # write-then-rename: an interrupted run never leaves half a file
    buf.tofile(str(tmp))
    os.replace(tmp, path)


def imwrite_rgb(path, rgb, quality=95):
    params = [cv2.IMWRITE_JPEG_QUALITY, quality] if str(path).lower().endswith((".jpg", ".jpeg")) else None
    imwrite_any(path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), params)


def save_json(obj, path):
    """Atomic JSON write (a crash never leaves a half-written file)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_json_default)
    os.replace(tmp, path)


def torch_save_atomic(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, tuple)):
        return list(o)
    if isinstance(o, Path):
        return str(o)
    return str(o)


def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s)).strip("_") or "x"


# ----------------------------------------------------------------------------- misc
def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def amp_dtype():
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def fmt_time(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


class Timer:
    def __init__(self):
        self.t0 = time.time()

    def __call__(self):
        return time.time() - self.t0


def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def worker_init_fn(worker_id):
    """DataLoader worker init (must live in a module so Windows 'spawn' can pickle it)."""
    cv2.setNumThreads(0)
    seed = (torch.initial_seed() + worker_id) % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


# ----------------------------------------------------------------------------- geometry
def letterbox(img, size: int, pad_value=255, interp=cv2.INTER_AREA):
    """Resize keeping aspect ratio into a size x size canvas (centered). Returns out, scale, pad_x, pad_y."""
    h, w = img.shape[:2]
    scale = size / max(h, w)
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    resized = cv2.resize(img, (nw, nh), interpolation=interp)
    pad_x, pad_y = (size - nw) // 2, (size - nh) // 2
    shape = (size, size) + img.shape[2:]
    out = np.full(shape, pad_value, dtype=img.dtype)
    out[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized
    return out, scale, pad_x, pad_y


def boxes_orig_to_cache(boxes, scale, pad_x, pad_y):
    b = np.asarray(boxes, dtype=np.float32).reshape(-1, 5).copy()
    b[:, [0, 2]] = b[:, [0, 2]] * scale + pad_x
    b[:, [1, 3]] = b[:, [1, 3]] * scale + pad_y
    return b


def boxes_cache_to_orig(boxes, scale, pad_x, pad_y, w=None, h=None):
    b = np.asarray(boxes, dtype=np.float32).copy()
    if b.size == 0:
        return b.reshape(-1, b.shape[-1] if b.ndim == 2 else 4)
    b[:, [0, 2]] = (b[:, [0, 2]] - pad_x) / scale
    b[:, [1, 3]] = (b[:, [1, 3]] - pad_y) / scale
    if w is not None:
        b[:, [0, 2]] = b[:, [0, 2]].clip(0, w)
    if h is not None:
        b[:, [1, 3]] = b[:, [1, 3]].clip(0, h)
    return b


def box_iou(a, b):
    """IoU matrix between (N,4) and (M,4) arrays of x1,y1,x2,y2."""
    a = np.asarray(a, dtype=np.float32).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float32).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]).clip(0) * (a[:, 3] - a[:, 1]).clip(0)
    area_b = (b[:, 2] - b[:, 0]).clip(0) * (b[:, 3] - b[:, 1]).clip(0)
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-6)


def box_ioa(a, b):
    """Intersection over area of a: (N,4) vs (M,4) -> (N,M)."""
    a = np.asarray(a, dtype=np.float32).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float32).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]).clip(0) * (a[:, 3] - a[:, 1]).clip(0)
    return inter / np.maximum(area_a[:, None], 1e-6)


def square_region(box, factor, min_side=0.0):
    """Square region of side factor*max(w,h) centred on box."""
    x1, y1, x2, y2 = [float(v) for v in box[:4]]
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(max(x2 - x1, y2 - y1) * factor, min_side)
    return cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2


def crop_with_pad(img, region, out_size, pad_value=255):
    """Crop a (possibly out-of-bounds) float region and resize to out_size x out_size."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = region
    side_x, side_y = x2 - x1, y2 - y1
    if side_x <= 1 or side_y <= 1:
        return np.full((out_size, out_size) + img.shape[2:], pad_value, img.dtype)
    sx, sy = out_size / side_x, out_size / side_y
    # affine that maps region -> output canvas (handles padding automatically)
    M = np.array([[sx, 0, -x1 * sx], [0, sy, -y1 * sy]], dtype=np.float32)
    interp = cv2.INTER_AREA if (sx < 1 and sy < 1) else cv2.INTER_LINEAR
    border = pad_value if img.ndim == 2 else (pad_value,) * img.shape[2]
    if interp == cv2.INTER_AREA:
        # warpAffine does not support INTER_AREA: crop in-bounds part first, resize, then paste
        ix1, iy1 = int(math.floor(max(0, x1))), int(math.floor(max(0, y1)))
        ix2, iy2 = int(math.ceil(min(w, x2))), int(math.ceil(min(h, y2)))
        canvas = np.full((out_size, out_size) + img.shape[2:], pad_value, img.dtype)
        if ix2 <= ix1 or iy2 <= iy1:
            return canvas
        patch = img[iy1:iy2, ix1:ix2]
        ox1, oy1 = max(0, int(round((ix1 - x1) * sx))), max(0, int(round((iy1 - y1) * sy)))
        ow = max(1, min(out_size - ox1, int(round((ix2 - ix1) * sx))))
        oh = max(1, min(out_size - oy1, int(round((iy2 - iy1) * sy))))
        if ox1 >= out_size or oy1 >= out_size:
            return canvas
        canvas[oy1:oy1 + oh, ox1:ox1 + ow] = cv2.resize(patch, (ow, oh), interpolation=cv2.INTER_AREA)
        return canvas
    return cv2.warpAffine(img, M, (out_size, out_size), flags=interp,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=border)
