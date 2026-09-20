"""Tiny synthetic STCray look-alike (same folders / file formats) for smoke-testing the notebook."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .config import CLASS_NAMES
from .utils import imwrite_any

FOLDERS = [f"Class {i + 1}_{n}" for i, n in enumerate(CLASS_NAMES)] + ["Class 21_Multilabel Threat",
                                                                      "Class 22_Non Threat"]
ORGANIC = np.array([0.95, 0.62, 0.25])
INORGANIC = np.array([0.45, 0.80, 0.35])
METAL = np.array([0.30, 0.50, 0.95])
CLASS_MATERIAL = [ORGANIC, METAL, ORGANIC, METAL, METAL, METAL, METAL, INORGANIC, INORGANIC, INORGANIC,
                  METAL, METAL, INORGANIC, METAL, METAL, METAL, METAL, METAL, METAL, METAL]


def _blend(img, mask, color, density):
    t = 1.0 - density * (1.0 - color)            # transmission of the object
    img[mask] *= t
    return img


def _shape(cls, cx, cy, s, rng):
    k = 3 + (cls % 6)
    ang = np.linspace(0, 2 * np.pi, k, endpoint=False) + rng.uniform(0, np.pi)
    ar = 0.35 + 0.08 * (cls % 5)
    r = s * (0.6 + 0.4 * rng.random(k))
    pts = np.stack([cx + r * np.cos(ang), cy + ar * r * np.sin(ang)], 1)
    rot = rng.uniform(0, np.pi)
    R = np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])
    return ((pts - [cx, cy]) @ R.T + [cx, cy]).astype(np.float32)


def _bag(h, w, rng):
    img = np.ones((h, w, 3), np.float32)
    x1, y1 = rng.integers(10, w // 6), rng.integers(10, h // 6)
    x2, y2 = rng.integers(w * 5 // 6, w - 10), rng.integers(h * 5 // 6, h - 10)
    m = np.zeros((h, w), np.uint8)
    cv2.rectangle(m, (int(x1), int(y1)), (int(x2), int(y2)), 1, -1)
    img = _blend(img, m > 0, ORGANIC, 0.25)
    for _ in range(rng.integers(6, 14)):
        mm = np.zeros((h, w), np.uint8)
        cx, cy = rng.integers(x1, x2), rng.integers(y1, y2)
        cv2.ellipse(mm, (int(cx), int(cy)), (int(rng.integers(8, 50)), int(rng.integers(5, 30))),
                    float(rng.uniform(0, 180)), 0, 360, 1, -1)
        mat = [ORGANIC, INORGANIC, METAL][rng.integers(0, 3)]
        img = _blend(img, mm > 0, mat, rng.uniform(0.2, 0.6))
    return img, (x1, y1, x2, y2)


def _threat(img, cls, bag, rng, s=None):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = bag
    s = s or rng.uniform(14, 34)
    cx, cy = rng.uniform(x1 + s, x2 - s), rng.uniform(y1 + s, y2 - s)
    pts = _shape(cls, cx, cy, s, rng)
    m = np.zeros((h, w), np.uint8)
    cv2.fillPoly(m, [np.round(pts).astype(np.int32)], 1)
    img = _blend(img, m > 0, CLASS_MATERIAL[cls], 0.85)
    ys, xs = np.nonzero(m)
    box = [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]
    return img, m, pts, box


def _labelme(shapes, w, h, name, rect):
    return dict(version="5.2.1", flags={}, imagePath=name, imageData=None, imageHeight=h, imageWidth=w,
                shapes=[dict(label=lab, points=([[b[0], b[1]], [b[2], b[3]]] if rect else p.tolist()),
                             group_id=None, shape_type="rectangle" if rect else "polygon", flags={})
                        for lab, p, b in shapes])


def _coco(shapes, w, h, name):
    cats = sorted({lab for lab, _, _ in shapes})
    cid = {c: i + 1 for i, c in enumerate(cats)}
    return dict(images=[dict(id=1, file_name=name, width=w, height=h)],
                categories=[dict(id=cid[c], name=c) for c in cats],
                annotations=[dict(id=k + 1, image_id=1, category_id=cid[lab], iscrowd=0,
                                  bbox=[b[0], b[1], b[2] - b[0], b[3] - b[1]], area=(b[2] - b[0]) * (b[3] - b[1]),
                                  segmentation=[p.flatten().tolist()])
                             for k, (lab, p, b) in enumerate(shapes)])


def make_fake_stcray(root, n_train=6, n_test=3, n_multi=(8, 4), n_benign=(6, 3), seed=0, bb_style="labelme",
                     label_style="name"):
    """Write a miniature STCray clone. bb_style: 'labelme' | 'coco'. label_style: 'name' | 'synonym'."""
    rng = np.random.default_rng(seed)
    root = Path(root)
    for split, n_per, nm, nb in (("STCray_TrainSet", n_train, n_multi[0], n_benign[0]),
                                 ("STCray_TestSet", n_test, n_multi[1], n_benign[1])):
        sd = root / split
        caps = {}
        for fi, folder in enumerate(FOLDERS):
            fid = fi + 1
            count = n_per if fid <= 20 else (nm if fid == 21 else nb)
            rows = []
            for k in range(count):
                h, w = int(rng.integers(260, 360)), int(rng.integers(360, 520))
                name = f"{folder.split('_', 1)[1].replace(' ', '')}_{k:04d}"
                img, bag = _bag(h, w, rng)
                shapes, full = [], np.zeros((h, w), np.uint8)
                if fid <= 20:
                    classes = [fid - 1]
                elif fid == 21:
                    classes = list(rng.choice(20, size=int(rng.integers(2, 4)), replace=False))
                else:
                    classes = []
                for c in classes:
                    img, m, pts, box = _threat(img, int(c), bag, rng)
                    full |= m
                    lab = CLASS_NAMES[c]
                    if label_style == "synonym":
                        lab = {"Injection": "syringe", "3D Gun": "3D printed gun"}.get(lab, lab.lower())
                    shapes.append((lab, pts, box))
                rgb = (np.clip(img, 0, 1) * 255).astype(np.uint8)
                rgb = cv2.GaussianBlur(rgb, (3, 3), 0)
                imwrite_any(sd / "Images" / folder / f"{name}.jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                            [cv2.IMWRITE_JPEG_QUALITY, 92])
                if fid <= 21:
                    imwrite_any(sd / "Segmentation" / folder / f"{name}.png", full * 255)
                    (sd / "Json" / folder).mkdir(parents=True, exist_ok=True)
                    (sd / "Json_BB" / folder).mkdir(parents=True, exist_ok=True)
                    with open(sd / "Json" / folder / f"{name}.json", "w") as f:
                        json.dump(_labelme(shapes, w, h, f"{name}.jpg", rect=False), f)
                    bb = _labelme(shapes, w, h, f"{name}.jpg", rect=True) if bb_style == "labelme" else \
                        _coco(shapes, w, h, f"{name}.jpg")
                    with open(sd / "Json_BB" / folder / f"{name}.json", "w") as f:
                        json.dump(bb, f)
                labs = ", ".join(s[0] for s in shapes) or "no prohibited items"
                clutter = ["limited", "medium", "heavy", "extreme"][int(rng.integers(0, 4))]
                rows.append((f"{name}.jpg", f"An X-ray scan of a bag with {clutter} clutter containing {labs}, "
                                            f"partially concealed beneath other items."))
                if fid <= 20 and k == 0:   # a near-duplicate re-scan of the same bag (threat moved)
                    img2, bag2 = img.copy(), bag
                    rgb2 = (np.clip(img2, 0, 1) * 255).astype(np.uint8)
                    rgb2 = cv2.GaussianBlur(rgb2, (3, 3), 0)
                    name2 = name + "_rescan"
                    imwrite_any(sd / "Images" / folder / f"{name2}.jpg", cv2.cvtColor(rgb2, cv2.COLOR_RGB2BGR),
                                [cv2.IMWRITE_JPEG_QUALITY, 92])
                    imwrite_any(sd / "Segmentation" / folder / f"{name2}.png", full * 255)
                    for sub, rect in (("Json", False), ("Json_BB", True)):
                        with open(sd / sub / folder / f"{name2}.json", "w") as f:
                            js = _labelme(shapes, w, h, f"{name2}.jpg", rect=rect) if (bb_style == "labelme" or
                                                                                       sub == "Json") \
                                else _coco(shapes, w, h, f"{name2}.jpg")
                            json.dump(js, f)
                    rows.append((f"{name2}.jpg", rows[-1][1]))
            caps[folder] = rows
        (sd / "Captions").mkdir(parents=True, exist_ok=True)
        for folder, rows in caps.items():
            pd.DataFrame(rows, columns=["Image Name", "Caption"]).to_excel(sd / "Captions" / f"{folder}.xlsx",
                                                                        index=False)
    (root / "_done.flag").write_text("ok")   # marks a complete build; an interrupted run leaves this missing
    return root


__all__ = ["make_fake_stcray", "FOLDERS"]
