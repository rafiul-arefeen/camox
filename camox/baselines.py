"""External baselines: YOLOv8 detector (Ultralytics) and zero-shot CLIP (open_clip). Both optional."""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .cache import cache_paths
from .config import CLASS_NAMES, NUM_CLASSES
from .utils import imread_rgb, load_json, save_json, worker_init_fn


# ----------------------------------------------------------------------------- YOLO
def export_yolo(splits: dict, cache_dir, out_dir, size=512):
    """splits: {'train': df, 'val': df, 'test': df}. Uses hard links to the 512px cache (no extra disk)."""
    out_dir = Path(out_dir)
    img_dir, _ = cache_paths(cache_dir, size)
    for split, d in splits.items():
        io, lo = out_dir / "images" / split, out_dir / "labels" / split
        io.mkdir(parents=True, exist_ok=True)
        lo.mkdir(parents=True, exist_ok=True)
        for r in d.itertuples(index=False):
            src, dst = img_dir / f"{r.uid}.jpg", io / f"{r.uid}.jpg"
            if not dst.exists():
                try:
                    os.link(src, dst)
                except OSError:
                    shutil.copy2(src, dst)
            lines = []
            for x1, y1, x2, y2, c in (r.boxes_cache or []):
                if c < 0:
                    continue
                cx, cy = (x1 + x2) / 2 / size, (y1 + y2) / 2 / size
                w, h = (x2 - x1) / size, (y2 - y1) / size
                lines.append(f"{int(c)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            (lo / f"{r.uid}.txt").write_text("\n".join(lines))
    yaml = out_dir / "stcray.yaml"
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(CLASS_NAMES))
    yaml.write_text(f"path: {out_dir.resolve().as_posix()}\ntrain: images/train\nval: images/val\n"
                    f"test: images/test\nnames:\n{names}\n")
    return yaml


def train_yolo(yaml, run_dir, model="yolov8s.pt", epochs=30, imgsz=512, batch=16, workers=4, seed=0, **extra):
    from ultralytics import YOLO
    run_dir = Path(run_dir)
    done = run_dir / "done.json"
    if done.exists():
        print(f"[{run_dir.name}] already trained - skipping")
        return run_dir / "weights" / "best.pt"
    last = run_dir / "weights" / "last.pt"
    if last.exists():
        YOLO(str(last)).train(resume=True)
    else:
        args = dict(data=str(yaml), epochs=epochs, imgsz=imgsz, batch=batch, workers=workers, seed=seed,
                    project=str(run_dir.parent), name=run_dir.name, exist_ok=True, plots=False, cos_lr=True,
                    close_mosaic=min(5, epochs), hsv_h=0.0, hsv_s=0.0, hsv_v=0.2, degrees=0.0, fliplr=0.5,
                    flipud=0.5, patience=max(10, epochs))
        args.update(extra)
        YOLO(model).train(**args)
    best = run_dir / "weights" / "best.pt"
    save_json(dict(weights=str(best)), done)
    return best


def yolo_predict(weights, df, cache_dir, size=512, conf=0.001, batch=32, device=None):
    """Image-level scores (max box confidence per class) + boxes in ORIGINAL coordinates."""
    from ultralytics import YOLO
    model = YOLO(str(weights))
    img_dir, _ = cache_paths(cache_dir, size)
    P = np.zeros((len(df), NUM_CLASSES), np.float32)
    boxes = {}
    rows = list(df.itertuples(index=False))
    paths = [str(img_dir / f"{r.uid}.jpg") for r in rows]
    for s in range(0, len(paths), batch):
        res = model.predict(paths[s:s + batch], imgsz=size, conf=conf, verbose=False, device=device,
                            max_det=100)
        for j, rr in enumerate(res):
            i = s + j
            r = rows[i]
            if rr.boxes is None or len(rr.boxes) == 0:
                continue
            xyxy = rr.boxes.xyxy.cpu().numpy()
            cls = rr.boxes.cls.cpu().numpy().astype(int)
            cf = rr.boxes.conf.cpu().numpy()
            np.maximum.at(P[i], cls, cf)
            xyxy[:, [0, 2]] = ((xyxy[:, [0, 2]] - r.pad_x) / r.scale).clip(0, r.img_w)
            xyxy[:, [1, 3]] = ((xyxy[:, [1, 3]] - r.pad_y) / r.scale).clip(0, r.img_h)
            boxes[r.uid] = [list(map(float, b)) + [int(c), float(p)] for b, c, p in zip(xyxy, cls, cf)]
    return P, boxes


# ----------------------------------------------------------------------------- CLIP
CLIP_NAMES = {"Explosive": "improvised explosive device", "3D Gun": "3D-printed gun", "Injection": "syringe",
              "Other Sharp Item": "sharp object", "Nail Cutter": "nail clipper", "Powerbank": "power bank",
              "Shaving Razor": "shaving razor", "Handcuffs": "pair of handcuffs"}
CLIP_TEMPLATES = ["an x-ray scan of a bag containing a {}.", "a baggage x-ray image with a {} inside.",
                  "a pseudo-colored airport security x-ray showing a {}.", "an x-ray image of a {}."]


class _ClipImages(Dataset):
    def __init__(self, paths, preprocess):
        self.paths, self.preprocess = list(paths), preprocess

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        from PIL import Image
        img = imread_rgb(self.paths[i])
        if img is None:
            img = np.full((224, 224, 3), 255, np.uint8)
        return self.preprocess(Image.fromarray(img)), i


@torch.no_grad()
def clip_zero_shot(df, cache_dir, size=512, model_name="ViT-B-16", pretrained="laion2b_s34b_b88k",
                   device=None, batch=64, num_workers=0):
    import open_clip
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    tok = open_clip.get_tokenizer(model_name)
    model = model.to(device).eval()
    names = [CLIP_NAMES.get(n, n.lower()) for n in CLASS_NAMES]
    prompts = names + ["harmless everyday item"]
    txt = []
    for n in prompts:
        t = model.encode_text(tok([tpl.format(n) for tpl in CLIP_TEMPLATES]).to(device)).float()
        t = t / t.norm(dim=-1, keepdim=True)
        t = t.mean(0)
        txt.append(t / t.norm())
    T = torch.stack(txt)
    img_dir, _ = cache_paths(cache_dir, size)
    ds = _ClipImages([img_dir / f"{u}.jpg" for u in df["uid"]], preprocess)
    dl = DataLoader(ds, batch_size=batch, num_workers=num_workers, worker_init_fn=worker_init_fn)
    probs, bag = [], []
    for x, _ in dl:
        f = model.encode_image(x.to(device)).float()
        f = f / f.norm(dim=-1, keepdim=True)
        logits = 100.0 * f @ T.T
        sm = logits.softmax(-1)
        probs.append(sm[:, :NUM_CLASSES].cpu().numpy())
        bag.append((1 - sm[:, -1]).cpu().numpy())
    return np.concatenate(probs), np.concatenate(bag)


__all__ = ["export_yolo", "train_yolo", "yolo_predict", "clip_zero_shot"]
