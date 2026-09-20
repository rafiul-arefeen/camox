"""Scan an STCray copy on disk, parse its annotations (format-sniffing) and build one index table.

Folder layout expected (both splits):
    <split>/Images/<Class k_Name>/*.jpg
    <split>/Json/<Class k_Name>/*.json        threat masks (polygons)
    <split>/Json_BB/<Class k_Name>/*.json     bounding boxes
    <split>/Segmentation/<Class k_Name>/*.png ground-truth masks
    <split>/Captions/*.xlsx                   image name + caption
"""
from __future__ import annotations

import base64
import difflib
import json
import math
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .config import (BENIGN_FOLDER_ID, CLASS_NAMES, GENERIC_LABELS, IGNORE_LABELS,
                     MULTI_FOLDER_ID, NUM_CLASSES, SYNONYMS)
from .utils import IMG_EXTS, safe_name

IMAGE_DIR_NAMES = ["Images", "Image", "images", "imgs"]
POLY_DIR_NAMES = ["Json", "JSON", "json", "Json_Mask", "Json_Masks"]
BOX_DIR_NAMES = ["Json_BB", "JSON_BB", "json_bb", "Json_BBox", "BB", "BBox"]
SEG_DIR_NAMES = ["Segmentation", "Segmentations", "segmentation", "Masks", "Mask", "GT"]
CAP_DIR_NAMES = ["Captions", "Caption", "captions"]
MASK_SUFFIXES = ("_mask", "-mask", "_seg", "-seg", "_gt", "_label", "_segmentation", "_bb", "_bbox")
SHAPE_WORDS = {"rectangle", "rect", "polygon", "polyline", "line", "linestrip", "circle", "point",
               "ellipse", "bbox", "box", "mask", "points"}


# ----------------------------------------------------------------------------- folders
def parse_class_folder(name: str):
    m = re.match(r"^\s*class\s*[_\- ]*(\d+)\s*[_\-. ]*(.*)$", str(name), flags=re.I)
    if not m:
        return None, None
    return int(m.group(1)), m.group(2).strip()


def _find_subdir(parent: Path, names):
    parent = Path(parent)
    if not parent.is_dir():
        return None
    children = {c.name.lower(): c for c in parent.iterdir() if c.is_dir()}
    for n in names:
        if n.lower() in children:
            return children[n.lower()]
    return None


def find_split_dirs(data_root=None, train_dir=None, test_dir=None):
    """Locate STCray_TrainSet / STCray_TestSet (tolerates doubled folders from unzipping)."""
    if train_dir and test_dir:
        out = {"train": Path(train_dir), "test": Path(test_dir)}
    else:
        root = Path(data_root)
        if not root.exists():
            raise FileNotFoundError(f"DATA_ROOT does not exist: {root}")
        cands = [root] + [p for p in root.glob("*") if p.is_dir()] + [p for p in root.glob("*/*") if p.is_dir()] \
            + [p for p in root.glob("*/*/*") if p.is_dir()]
        cands = [p for p in cands if _find_subdir(p, IMAGE_DIR_NAMES) is not None]
        pick = {}
        for split in ("train", "test"):
            hits = [p for p in cands if split in p.name.lower()]
            if not hits:
                raise FileNotFoundError(
                    f"Could not find a '{split}' folder containing 'Images' under {root}. "
                    f"Set TRAIN_DIR / TEST_DIR explicitly.")
            pick[split] = sorted(hits, key=lambda p: len(p.parts))[0]
        out = pick
    for k, v in out.items():
        if _find_subdir(v, IMAGE_DIR_NAMES) is None:
            raise FileNotFoundError(f"{k} folder {v} has no Images/ sub-folder")
    return out


def _stem_key(p: Path):
    s = p.stem.lower()
    for suf in MASK_SUFFIXES:
        if s.endswith(suf):
            s = s[: -len(suf)]
            break
    return s


def _lookup(ann_root, folder_name, exts):
    """stem -> path for one annotation folder (class sub-folder, searched recursively)."""
    if ann_root is None:
        return {}
    d = ann_root / folder_name
    if not d.is_dir():
        # tolerate small naming differences between Images/ and annotation class folders
        fid, _ = parse_class_folder(folder_name)
        d = None
        for c in ann_root.iterdir():
            if c.is_dir() and parse_class_folder(c.name)[0] == fid:
                d = c
                break
        if d is None:
            return {}
    out = {}
    for p in d.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            out.setdefault(p.stem.lower(), p)
            out.setdefault(_stem_key(p), p)
    return out


# ----------------------------------------------------------------------------- labels
def normalize_label(raw):
    """Map a raw annotation label to a class index. Returns (index or None, status)."""
    if raw is None:
        return None, "missing"
    s = str(raw).strip().lower()
    m = re.match(r"^class\s*[_\- ]*(\d+)\s*[_\-: .]*(.*)$", s)
    if m:
        if not m.group(2):
            v = int(m.group(1))
            return (v - 1, "numeric") if 1 <= v <= NUM_CLASSES else (None, "ignore" if v == BENIGN_FOLDER_ID else "unmapped")
        s = m.group(2)
    key = re.sub(r"[^a-z0-9]", "", s)
    if key in IGNORE_LABELS:
        return None, "ignore"
    if key in GENERIC_LABELS:
        return None, "generic"
    if key in SYNONYMS:
        return SYNONYMS[key], "ok"
    if key.isdigit():
        v = int(key)
        return (v - 1, "numeric") if 1 <= v <= NUM_CLASSES else (None, "unmapped")
    if key.startswith("3d") or "3dprint" in key or "printedgun" in key:
        return SYNONYMS["3dgun"], "fuzzy"
    if key.startswith("ied") or "explos" in key or "bomb" in key:
        return SYNONYMS["explosive"], "fuzzy"
    best = max((k for k in SYNONYMS if len(k) >= 4 and k in key), key=len, default=None)
    if best:
        return SYNONYMS[best], "fuzzy"
    close = difflib.get_close_matches(key, list(SYNONYMS.keys()), n=1, cutoff=0.8)
    if close:
        return SYNONYMS[close[0]], "fuzzy"
    return None, "unmapped"


# ----------------------------------------------------------------------------- JSON parsing
def _is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))


def _ci_get(d, key):
    if key in d:
        return d[key]
    kl = key.lower()
    for k, v in d.items():
        if isinstance(k, str) and k.lower() == kl:
            return v
    return None


def _pts_array(points):
    try:
        if points is None or len(points) == 0:
            return None
        first = points[0]
        if isinstance(first, dict):
            arr = [[float(_ci_get(p, "x")), float(_ci_get(p, "y"))] for p in points]
        elif isinstance(first, (list, tuple)):
            if len(first) and isinstance(first[0], (list, tuple)):  # nested polygon list -> first polygon
                arr = [[float(q[0]), float(q[1])] for q in first]
            else:
                arr = [[float(p[0]), float(p[1])] for p in points if len(p) >= 2]
        elif _is_num(first):
            flat = [float(v) for v in points]
            arr = list(zip(flat[0::2], flat[1::2]))
        else:
            return None
        a = np.asarray(arr, dtype=np.float32).reshape(-1, 2)
        return a if len(a) else None
    except Exception:
        return None


def _bbox_from_pts(pts):
    return [float(pts[:, 0].min()), float(pts[:, 1].min()), float(pts[:, 0].max()), float(pts[:, 1].max())]


def _bbox_from_keys(d):
    keys = {str(k).lower().replace("_", ""): v for k, v in d.items() if isinstance(k, str)}

    def num(*names):
        for n in names:
            v = keys.get(n)
            if _is_num(v):
                return float(v)
        return None

    xmin, ymin, xmax, ymax = num("xmin", "minx", "x1", "left"), num("ymin", "miny", "y1", "top"), \
        num("xmax", "maxx", "x2", "right"), num("ymax", "maxy", "y2", "bottom")
    if None not in (xmin, ymin, xmax, ymax):
        return [xmin, ymin, xmax, ymax]
    w, h = num("width", "w"), num("height", "h")
    cx, cy = num("cx", "xc", "centerx", "xcenter"), num("cy", "yc", "centery", "ycenter")
    if None not in (cx, cy, w, h):
        return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]
    x, y = num("x", "left", "xmin", "x1"), num("y", "top", "ymin", "y1")
    if None not in (x, y, w, h):
        return [x, y, x + w, y + h]
    return None


def _geometry(d):
    """Return {'bbox': xyxy|None, 'bbox_raw': [4]|None, 'polys': [arr]|None, 'mask_b64': str|None} or None."""
    # VGG Image Annotator
    sa = d.get("shape_attributes")
    if isinstance(sa, dict):
        name = str(sa.get("name", "")).lower()
        try:
            if name == "rect":
                x, y, w, h = [float(sa[k]) for k in ("x", "y", "width", "height")]
                return dict(bbox=[x, y, x + w, y + h], bbox_raw=None, polys=None, mask_b64=None)
            if name in ("polygon", "polyline"):
                pts = np.stack([np.asarray(sa["all_points_x"], np.float32),
                                np.asarray(sa["all_points_y"], np.float32)], 1)
                return dict(bbox=_bbox_from_pts(pts), bbox_raw=None, polys=[pts], mask_b64=None)
            if name == "circle":
                cx, cy, r = float(sa["cx"]), float(sa["cy"]), float(sa["r"])
                return dict(bbox=[cx - r, cy - r, cx + r, cy + r], bbox_raw=None, polys=None, mask_b64=None)
            if name == "ellipse":
                cx, cy, rx, ry = float(sa["cx"]), float(sa["cy"]), float(sa["rx"]), float(sa["ry"])
                return dict(bbox=[cx - rx, cy - ry, cx + rx, cy + ry], bbox_raw=None, polys=None, mask_b64=None)
        except Exception:
            return None
        return None
    # LabelMe / X-AnyLabeling / generic point lists
    if isinstance(d.get("points"), (list, tuple)):
        pts = _pts_array(d["points"])
        if pts is None or len(pts) < 2:
            return None
        st = str(d.get("shape_type", "")).lower()
        if st == "circle":
            c = pts[0]
            r = float(np.linalg.norm(pts[1] - pts[0]))
            return dict(bbox=[float(c[0] - r), float(c[1] - r), float(c[0] + r), float(c[1] + r)], bbox_raw=None,
                        polys=None, mask_b64=None)
        if st == "mask" and isinstance(d.get("mask"), str):
            return dict(bbox=_bbox_from_pts(pts), bbox_raw=None, polys=None, mask_b64=d["mask"])
        if st in ("rectangle", "rect", "bbox") or len(pts) == 2:
            return dict(bbox=_bbox_from_pts(pts), bbox_raw=None, polys=None, mask_b64=None)
        if st in ("point", "line"):
            return None
        return dict(bbox=_bbox_from_pts(pts), bbox_raw=None, polys=[pts], mask_b64=None)
    geom = None
    # COCO-style segmentation
    seg = d.get("segmentation")
    polys = None
    if isinstance(seg, list) and len(seg):
        if isinstance(seg[0], (list, tuple)):
            polys = [p for p in (_pts_array(s) for s in seg) if p is not None and len(p) >= 3]
        elif _is_num(seg[0]):
            p = _pts_array(seg)
            polys = [p] if p is not None and len(p) >= 3 else None
    coco_hint = any(k in d for k in ("category_id", "iscrowd", "area", "image_id"))
    for key in ("bbox", "box", "bndbox", "rect", "rectangle", "bounding_box", "boundingbox", "boundingBox", "bb"):
        v = _ci_get(d, key)
        if isinstance(v, (list, tuple)) and len(v) == 4 and all(_is_num(t) for t in v):
            vv = [float(t) for t in v]
            if coco_hint:
                geom = dict(bbox=[vv[0], vv[1], vv[0] + vv[2], vv[1] + vv[3]], bbox_raw=None)
            else:
                geom = dict(bbox=None, bbox_raw=vv)
            break
        if isinstance(v, (list, tuple)) and len(v) >= 2 and all(isinstance(t, (list, tuple, dict)) for t in v):
            pts = _pts_array(v)
            if pts is not None and len(pts) >= 2:
                geom = dict(bbox=_bbox_from_pts(pts), bbox_raw=None)
                break
        if isinstance(v, dict):
            bb = _bbox_from_keys(v)
            if bb is not None:
                geom = dict(bbox=bb, bbox_raw=None)
                break
    if geom is None:
        coords = d.get("coordinates")
        if isinstance(coords, dict):
            k = {str(a).lower(): b for a, b in coords.items()}
            if all(_is_num(k.get(n)) for n in ("x", "y", "width", "height")):  # CreateML: centre based
                x, y, w, h = [float(k[n]) for n in ("x", "y", "width", "height")]
                geom = dict(bbox=[x - w / 2, y - h / 2, x + w / 2, y + h / 2], bbox_raw=None)
    if geom is None:
        bb = _bbox_from_keys(d)
        if bb is not None:
            geom = dict(bbox=bb, bbox_raw=None)
    if geom is None and polys:
        allp = np.concatenate(polys, 0)
        geom = dict(bbox=_bbox_from_pts(allp), bbox_raw=None)
    if geom is None:
        return None
    geom["polys"] = polys
    geom["mask_b64"] = None
    return geom


_LABEL_KEYS = ("label", "name", "class", "class_name", "classname", "category", "category_name",
               "categoryname", "classTitle", "title", "tag", "cls", "object", "type")


def _label(d, cat_map):
    for k in _LABEL_KEYS:
        v = _ci_get(d, k)
        if isinstance(v, str) and v.strip() and v.strip().lower() not in SHAPE_WORDS:
            return v.strip()
        if _is_num(v) and k in ("label", "class", "cls", "category"):
            return str(int(v))
        if isinstance(v, dict):
            n = v.get("name") or v.get("label") or v.get("title")
            if isinstance(n, str) and n.strip():
                return n.strip()
    cid = d.get("category_id")
    if cid is not None:
        return str(cat_map.get(cid, cid))
    ra = d.get("region_attributes")
    if isinstance(ra, dict) and ra:
        for _, v in ra.items():
            if isinstance(v, str) and v.strip():
                return v.strip()
            if isinstance(v, dict):
                for kk, vv in v.items():
                    if vv:
                        return str(kk)
        return str(next(iter(ra.keys())))
    tags = d.get("tags")
    if isinstance(tags, list) and tags and isinstance(tags[0], str):
        return tags[0]
    return None


def _parallel_arrays(d, out, cat_map):
    """{'boxes': [[..],..], 'labels': [..]} style annotations."""
    box_key = next((k for k in ("boxes", "bboxes", "bbox", "rects") if isinstance(d.get(k), list)), None)
    lab_key = next((k for k in ("labels", "classes", "names", "categories", "label_names")
                    if isinstance(d.get(k), list)), None)
    if box_key is None or lab_key is None:
        return False
    boxes, labels = d[box_key], d[lab_key]
    if not boxes or not all(isinstance(b, (list, tuple)) and len(b) == 4 and all(_is_num(t) for t in b) for b in boxes):
        return False
    if len(labels) != len(boxes):
        return False
    for b, lab in zip(boxes, labels):
        if isinstance(lab, dict):
            lab = lab.get("name")
        lab = cat_map.get(lab, lab)
        out.append(dict(label=str(lab), bbox=None, bbox_raw=[float(t) for t in b], polys=None, mask_b64=None))
    return True


def _walk(o, out, cat_map, parent_label, depth):
    if depth > 12:
        return
    if isinstance(o, dict):
        if _parallel_arrays(o, out, cat_map):
            return
        geom = _geometry(o)
        if geom is not None:
            st = o.get("shape_type")
            if st is None and isinstance(o.get("shape_attributes"), dict):
                st = "via-" + str(o["shape_attributes"].get("name"))
            out.append(dict(label=_label(o, cat_map) or parent_label, shape_type=st, **geom))
            return
        here = _label(o, cat_map) if depth > 0 else None
        for k, v in o.items():
            if k in ("imageData", "image_data", "categories", "info", "licenses"):
                continue
            if isinstance(v, (dict, list)):
                _walk(v, out, cat_map, here or parent_label, depth + 1)
    elif isinstance(o, list):
        if depth == 0 and len(o) == 4 and all(_is_num(t) for t in o):
            out.append(dict(label=None, bbox=None, bbox_raw=[float(t) for t in o], polys=None, mask_b64=None))
            return
        for v in o:
            if isinstance(v, (dict, list)):
                _walk(v, out, cat_map, parent_label, depth + 1)


_STYLE_CACHE = {}


def _json_style(path):
    """'labelme' | 'coco' | 'via' | 'other' for one annotation file (only the first few are opened)."""
    if not path:
        return None
    if len(_STYLE_CACHE) > 200:
        return None
    js = load_json_file(path)
    style = "other"
    if isinstance(js, dict):
        if isinstance(js.get("shapes"), list) and ("imagePath" in js or "version" in js):
            style = "labelme"
        elif isinstance(js.get("annotations"), list) and isinstance(js.get("images"), list):
            style = "coco"
        elif any(isinstance(v, dict) and "regions" in v for v in js.values()):
            style = "via"
    _STYLE_CACHE[path] = style
    return style


def load_json_file(path):
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            with open(path, "r", encoding=enc) as f:
                return json.load(f)
        except UnicodeDecodeError:
            continue
        except Exception:
            return None
    return None


def extract_objects(js):
    """Generic annotation extractor. Returns (objects, meta)."""
    cat_map = {}
    meta = {}
    if isinstance(js, dict):
        cats = js.get("categories")
        if isinstance(cats, list):
            for c in cats:
                if isinstance(c, dict) and "id" in c:
                    cat_map[c["id"]] = c.get("name", str(c["id"]))
        for wk, hk in (("imageWidth", "imageHeight"), ("width", "height"), ("image_width", "image_height")):
            if _is_num(js.get(wk)) and _is_num(js.get(hk)):
                meta["w"], meta["h"] = float(js[wk]), float(js[hk])
                break
        if "w" not in meta and isinstance(js.get("images"), list) and len(js["images"]) == 1:
            im = js["images"][0]
            if isinstance(im, dict) and _is_num(im.get("width")) and _is_num(im.get("height")):
                meta["w"], meta["h"] = float(im["width"]), float(im["height"])
    out = []
    if js is not None:
        _walk(js, out, cat_map, None, 0)
    return out, meta


def describe_json(path, max_list=2, max_depth=5, max_str=60):
    """Readable outline of one annotation file (for the diagnostics cell)."""
    js = load_json_file(path)
    lines = []

    def rec(o, indent, depth):
        pad = "  " * indent
        if depth > max_depth:
            lines.append(pad + "...")
            return
        if isinstance(o, dict):
            for k, v in o.items():
                if isinstance(v, (dict, list)):
                    kind = f"list[{len(v)}]" if isinstance(v, list) else "dict"
                    lines.append(f"{pad}{k}: {kind}")
                    rec(v, indent + 1, depth + 1)
                else:
                    s = repr(v)
                    if len(s) > max_str:
                        s = s[:max_str] + f"... ({len(s)} chars)"
                    lines.append(f"{pad}{k}: {s}")
        elif isinstance(o, list):
            if len(o) and all(_is_num(t) for t in o):
                lines.append(pad + (repr(o) if len(o) <= 8 else repr(o[:8])[:-1] + ", ...]"))
                return
            for i, v in enumerate(o[:max_list]):
                lines.append(f"{pad}[{i}]")
                rec(v, indent + 1, depth + 1)
            if len(o) > max_list:
                lines.append(f"{pad}... ({len(o) - max_list} more)")
    if js is None:
        return f"<could not parse {path}>"
    rec(js, 0, 0)
    return "\n".join(lines)


def decode_mask_b64(b64):
    import cv2
    try:
        data = np.frombuffer(base64.b64decode(b64), np.uint8)
        m = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
        if m is None:
            return None
        if m.ndim == 3:
            m = m.max(axis=2)
        return (m > 0).astype(np.uint8)
    except Exception:
        return None


# ----------------------------------------------------------------------------- captions
def load_captions(split_dir):
    cap_dir = _find_subdir(split_dir, CAP_DIR_NAMES)
    by_folder, by_stem = {}, defaultdict(list)
    info = {"files": 0, "rows": 0}
    if cap_dir is None:
        return by_folder, by_stem, info
    for f in sorted(cap_dir.rglob("*")):
        if not f.is_file() or f.suffix.lower() not in (".xlsx", ".xls", ".csv"):
            continue
        try:
            df = pd.read_csv(f, header=None) if f.suffix.lower() == ".csv" else pd.read_excel(f, header=None)
        except Exception as e:  # noqa: BLE001
            info.setdefault("errors", []).append(f"{f.name}: {e}")
            continue
        df = df.dropna(how="all")
        if df.shape[1] < 2 or len(df) == 0:
            continue
        cols = list(df.columns[:2]) if df.shape[1] == 2 else list(df.columns)
        lens = {c: df[c].astype(str).str.len().mean() for c in cols}
        cap_col = max(lens, key=lens.get)
        name_col = min((c for c in cols if c != cap_col), key=lambda c: lens[c])
        first = str(df.iloc[0][name_col]).strip().lower()
        if re.fullmatch(r"(image|img|file|name|filename|image ?name|image_id|id)s?[_ ]?(name)?", first):
            df = df.iloc[1:]
        fid, _ = parse_class_folder(f.stem)
        info["files"] += 1
        for name, cap in zip(df[name_col].tolist(), df[cap_col].tolist()):
            if not isinstance(cap, str) or not str(name).strip():
                continue
            base = str(name).replace("\\", "/").split("/")[-1].strip()
            stem = base.rsplit(".", 1)[0].lower() if base.lower().endswith(IMG_EXTS) else base.lower()
            if fid is not None:
                by_folder[(fid, stem)] = cap
            by_stem[stem].append(cap)
            info["rows"] += 1
    return by_folder, by_stem, info


# ----------------------------------------------------------------------------- index
def _image_size(path):
    try:
        with Image.open(path) as im:
            return im.size  # (w, h)
    except Exception:
        return (None, None)


def _parse_one(item):
    rec = dict(item)
    w, h = _image_size(rec["img_path"])
    rec["img_w"], rec["img_h"] = w, h
    objs_box, objs_poly = [], []
    if rec.get("bb_path"):
        js = load_json_file(rec["bb_path"])
        if js is None:
            rec["_bb_error"] = True
        else:
            objs_box, meta = extract_objects(js)
            if rec["img_w"] is None and "w" in meta:
                rec["img_w"], rec["img_h"] = meta["w"], meta["h"]
    if rec.get("json_path"):
        js = load_json_file(rec["json_path"])
        if js is None:
            rec["_json_error"] = True
        else:
            objs_poly, _ = extract_objects(js)
    # keep polygons only as a flag (they are re-read from disk if a mask must be rasterised)
    rec["_objs_box"] = [dict(label=o["label"], bbox=o["bbox"], bbox_raw=o["bbox_raw"],
                             shape_type=o.get("shape_type")) for o in objs_box]
    rec["_objs_poly"] = [dict(label=o["label"], bbox=o["bbox"], bbox_raw=o["bbox_raw"],
                              has_poly=bool(o.get("polys")) or o.get("mask_b64") is not None,
                              shape_type=o.get("shape_type"))
                         for o in objs_poly]
    rec["_bb_style"] = _json_style(rec.get("bb_path"))

    return rec


def _resolve_box(o, fmt, w, h):
    b = o.get("bbox")
    if b is None and o.get("bbox_raw") is not None:
        r = o["bbox_raw"]
        b = [r[0], r[1], r[0] + r[2], r[1] + r[3]] if fmt == "xywh" else list(r)
    if b is None:
        return None
    b = [float(v) for v in b]
    if w and h and max(abs(v) for v in b) <= 1.5:  # normalised coordinates
        b = [b[0] * w, b[1] * h, b[2] * w, b[3] * h]
    x1, x2 = sorted((b[0], b[2]))
    y1, y2 = sorted((b[1], b[3]))
    if w and h:
        x1, x2 = max(0.0, x1), min(float(w), x2)
        y1, y2 = max(0.0, y1), min(float(h), y2)
    if x2 - x1 < 1 or y2 - y1 < 1:
        return None
    return [x1, y1, x2, y2]


def detect_bbox_format(records, max_n=3000):
    raws, pairs = [], []
    for r in records:
        amb = [o["bbox_raw"] for o in r["_objs_box"] if o.get("bbox") is None and o.get("bbox_raw") is not None]
        if not amb:
            continue
        raws.extend(amb)
        polys = [o["bbox"] for o in r["_objs_poly"] if o.get("bbox") is not None]
        if polys and len(amb) == 1 and len(polys) >= 1:
            pb = np.asarray(polys, np.float32)
            pairs.append((amb[0], [pb[:, 0].min(), pb[:, 1].min(), pb[:, 2].max(), pb[:, 3].max()]))
        if len(raws) >= max_n:
            break
    if not raws:
        return "xyxy", "no ambiguous 4-number boxes found (formats were explicit)"
    arr = np.asarray(raws, np.float32)
    bad = int(((arr[:, 2] < arr[:, 0]) | (arr[:, 3] < arr[:, 1])).sum())
    if bad:
        return "xywh", f"{bad}/{len(arr)} boxes have x2<x1 or y2<y1 -> [x, y, w, h]"
    if pairs:
        from .utils import box_iou
        a = np.asarray([p[0] for p in pairs], np.float32)
        g = np.asarray([p[1] for p in pairs], np.float32)
        xyxy = np.diag(box_iou(a, g)).mean()
        a2 = a.copy()
        a2[:, 2] += a2[:, 0]
        a2[:, 3] += a2[:, 1]
        xywh = np.diag(box_iou(a2, g)).mean()
        fmt = "xyxy" if xyxy >= xywh else "xywh"
        return fmt, f"agreement with mask polygons: IoU xyxy={xyxy:.3f} vs xywh={xywh:.3f} ({len(pairs)} images)"
    return "xyxy", f"all {len(arr)} boxes consistent with [x1, y1, x2, y2] (no polygons to cross-check)"


def index_split(split, split_dir, workers=8, progress=True):
    split_dir = Path(split_dir)
    img_root = _find_subdir(split_dir, IMAGE_DIR_NAMES)
    poly_root = _find_subdir(split_dir, POLY_DIR_NAMES)
    box_root = _find_subdir(split_dir, BOX_DIR_NAMES)
    seg_root = _find_subdir(split_dir, SEG_DIR_NAMES)
    items, skipped = [], Counter()
    folders = sorted([d for d in img_root.iterdir() if d.is_dir()], key=lambda d: (parse_class_folder(d.name)[0] or 999, d.name))
    for d in folders:
        fid, fname = parse_class_folder(d.name)
        if fid is None or not (1 <= fid <= BENIGN_FOLDER_ID):
            skipped[d.name] += 1
            continue
        lk_poly = _lookup(poly_root, d.name, (".json",))
        lk_box = _lookup(box_root, d.name, (".json",))
        lk_seg = _lookup(seg_root, d.name, IMG_EXTS)
        for p in sorted(d.rglob("*")):
            if not (p.is_file() and p.suffix.lower() in IMG_EXTS):
                continue
            key = p.stem.lower()
            items.append(dict(split=split, folder_id=fid, folder_name=d.name, img_path=str(p), stem=p.stem,
                              json_path=str(lk_poly[key]) if key in lk_poly else None,
                              bb_path=str(lk_box[key]) if key in lk_box else None,
                              seg_path=str(lk_seg[key]) if key in lk_seg else None))
    iterator = items
    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = ex.map(_parse_one, iterator)
        if progress:
            try:
                from tqdm.auto import tqdm
                results = tqdm(results, total=len(items), desc=f"parse {split}")
            except Exception:
                pass
        recs = list(results)
    info = dict(split_dir=str(split_dir), images_dir=str(img_root),
                json_dir=str(poly_root) if poly_root else None,
                json_bb_dir=str(box_root) if box_root else None,
                seg_dir=str(seg_root) if seg_root else None,
                skipped_folders=dict(skipped))
    return recs, info


def finalize_records(recs, fmt, captions):
    vocab = Counter()
    issues = Counter()
    conflicts = Counter()
    rows = []
    used_uids = set()
    for r in recs:
        fid = r["folder_id"]
        w, h = r.get("img_w"), r.get("img_h")
        iss = []
        if w is None:
            iss.append("img_unreadable")
        boxes = []
        for src in ("_objs_box", "_objs_poly"):
            objs = r[src]
            if src == "_objs_poly" and boxes:
                # polygons only add labels / fallback boxes when Json_BB had none
                for o in objs:
                    cls, st = normalize_label(o["label"])
                    vocab[(str(o["label"]), CLASS_NAMES[cls] if cls is not None else None, st)] += 1
                continue
            for o in objs:
                cls, st = normalize_label(o["label"])
                vocab[(str(o["label"]), CLASS_NAMES[cls] if cls is not None else None, st)] += 1
                if st == "ignore":
                    continue
                b = _resolve_box(o, fmt, w, h)
                if b is None:
                    continue
                boxes.append([*b, -1 if cls is None else cls, st])
        # ----- class assignment rules
        if fid == BENIGN_FOLDER_ID:
            if boxes:
                iss.append("benign_with_objects")
            labels, final_boxes = [], []
            label_source = "benign"
        elif fid == MULTI_FOLDER_ID:
            final_boxes = [b[:5] for b in boxes if b[4] >= 0]
            if any(b[4] < 0 for b in boxes):
                iss.append("unmapped_label")
            poly_cls = {normalize_label(o["label"])[0] for o in r["_objs_poly"]}
            labels = sorted({int(b[4]) for b in final_boxes} | {c for c in poly_cls if c is not None})
            if not labels:
                iss.append("multi_no_labels")
            label_source = "annotation"
        else:
            c = fid - 1
            for b in boxes:
                if b[4] >= 0 and b[4] != c and b[5] in ("ok", "numeric"):
                    conflicts[(CLASS_NAMES[c], CLASS_NAMES[b[4]])] += 1
            final_boxes = [[*b[:4], c] for b in boxes]
            labels = [c]
            label_source = "folder"
        if fid != BENIGN_FOLDER_ID and not final_boxes:
            iss.append("no_boxes")
        if r.get("bb_path") is None and fid != BENIGN_FOLDER_ID:
            iss.append("no_json_bb")
        if r.get("seg_path") is None and fid != BENIGN_FOLDER_ID:
            iss.append("no_seg_png")
        has_poly = any(o.get("has_poly") for o in r["_objs_poly"])
        mask_source = "png" if r.get("seg_path") else ("poly" if has_poly else ("box" if final_boxes else "none"))
        if fid == BENIGN_FOLDER_ID:
            mask_source = "png" if r.get("seg_path") else "none"
        stem_l = r["stem"].lower()
        cap = captions["by_folder"].get((fid, stem_l))
        if cap is None and len(captions["by_stem"].get(stem_l, [])) == 1:
            cap = captions["by_stem"][stem_l][0]
        uid = f"{r['split']}_{fid:02d}_{safe_name(r['stem'])}"
        k = 1
        while uid in used_uids:
            k += 1
            uid = f"{r['split']}_{fid:02d}_{safe_name(r['stem'])}_{k}"
        used_uids.add(uid)
        for i in iss:
            issues[i] += 1
        rows.append(dict(
            uid=uid, split=r["split"], folder_id=fid, folder_name=r["folder_name"], stem=r["stem"],
            img_path=r["img_path"], json_path=r.get("json_path"), bb_path=r.get("bb_path"),
            seg_path=r.get("seg_path"), img_w=w, img_h=h, labels=labels,
            n_labels=len(labels), is_threat=len(labels) > 0, boxes=final_boxes, n_boxes=len(final_boxes),
            label_source=label_source, mask_source=mask_source, caption=cap, issues=";".join(iss),
        ))
    return rows, vocab, issues, conflicts


def build_index(data_root=None, train_dir=None, test_dir=None, workers=8, bbox_format="auto", progress=True):
    _STYLE_CACHE.clear()
    dirs = find_split_dirs(data_root, train_dir, test_dir)
    all_recs, info = [], {}
    caps = {}
    for split, sdir in dirs.items():
        recs, inf = index_split(split, sdir, workers, progress)
        by_folder, by_stem, cinfo = load_captions(sdir)
        inf["captions"] = cinfo
        caps[split] = dict(by_folder=by_folder, by_stem=by_stem)
        info[split] = inf
        all_recs.extend(recs)
    if bbox_format == "auto":
        fmt, why = detect_bbox_format(all_recs)
    else:
        fmt, why = bbox_format, "set by user"
    rows = []
    vocab, issues, conflicts = Counter(), Counter(), Counter()
    for split in dirs:
        sr = [r for r in all_recs if r["split"] == split]
        rws, v, i, c = finalize_records(sr, fmt, caps[split])
        rows.extend(rws)
        vocab.update(v)
        issues.update(i)
        conflicts.update(c)
    df = pd.DataFrame(rows)
    shape_types = dict(Json_BB=Counter(o.get("shape_type") for r in all_recs for o in r["_objs_box"]),
                       Json=Counter(o.get("shape_type") for r in all_recs for o in r["_objs_poly"]))
    styles = Counter(r.get("_bb_style") for r in all_recs if r.get("_bb_style"))
    diag = dict(dirs={k: str(v) for k, v in dirs.items()}, info=info, bbox_format=fmt, bbox_format_reason=why,
                shape_types=shape_types, json_styles=styles,
                label_vocab=vocab, issues=issues, conflicts=conflicts,
                bb_parse_errors=sum(1 for r in all_recs if r.get("_bb_error")),
                json_parse_errors=sum(1 for r in all_recs if r.get("_json_error")))
    return df, diag


def label_matrix(df):
    y = np.zeros((len(df), NUM_CLASSES), dtype=np.float32)
    for i, labs in enumerate(df["labels"].tolist()):
        for c in labs:
            y[i, int(c)] = 1.0
    return y


def vocab_table(vocab):
    rows = [dict(raw_label=k[0], mapped_to=k[1], status=k[2], count=v) for k, v in vocab.items()]
    if not rows:
        return pd.DataFrame(columns=["raw_label", "mapped_to", "status", "count"])
    return pd.DataFrame(rows).sort_values("count", ascending=False).reset_index(drop=True)


# ----------------------------------------------------------------------------- polygons -> mask
def rasterize_annotations(json_path, w, h, fmt="xyxy", scale=1.0, offset=(0.0, 0.0), out_hw=None):
    """Rasterise polygon / LabelMe-mask annotations of one file into a uint8 {0,1} mask."""
    import cv2
    js = load_json_file(json_path) if json_path else None
    H, W = out_hw if out_hw else (int(round(h * scale)), int(round(w * scale)))
    mask = np.zeros((H, W), np.uint8)
    if js is None:
        return mask, 0
    n = 0
    objs = []
    _walk(js, objs, {}, None, 0)
    for o in objs:
        if normalize_label(o.get("label"))[1] == "ignore":
            continue
        if o.get("polys"):
            for p in o["polys"]:
                q = p * scale + np.asarray(offset, np.float32)
                cv2.fillPoly(mask, [np.round(q).astype(np.int32)], 1)
                n += 1
        elif o.get("mask_b64") and o.get("bbox"):
            m = decode_mask_b64(o["mask_b64"])
            if m is None:
                continue
            x1, y1, x2, y2 = o["bbox"]
            X1, Y1 = int(round(x1 * scale + offset[0])), int(round(y1 * scale + offset[1]))
            X2, Y2 = int(round(x2 * scale + offset[0])), int(round(y2 * scale + offset[1]))
            if X2 <= X1 or Y2 <= Y1:
                continue
            m = cv2.resize(m, (X2 - X1, Y2 - Y1), interpolation=cv2.INTER_NEAREST)
            xa, ya = max(0, X1), max(0, Y1)
            xb, yb = min(W, X2), min(H, Y2)
            if xb > xa and yb > ya:
                mask[ya:yb, xa:xb] |= m[ya - Y1:yb - Y1, xa - X1:xb - X1]
                n += 1
    return mask, n
