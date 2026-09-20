"""Training / inference engine: AMP, auto batch size, gradient accumulation, resume, run registry."""
from __future__ import annotations

import gc
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .evaluate import mean_ap
from .losses import build_cls_loss, seg_loss
from .models import build_model, count_params
from .utils import (amp_dtype, cuda_sync, fmt_time, get_device, load_json, save_json, set_seed,
                    torch_save_atomic, worker_init_fn)


def make_loader(ds, batch_size, shuffle, num_workers, drop_last=False):
    kw = dict(batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
              pin_memory=torch.cuda.is_available(), drop_last=drop_last, worker_init_fn=worker_init_fn)
    if num_workers > 0:
        kw.update(persistent_workers=True, prefetch_factor=2)
    return DataLoader(ds, **kw)


def _autocast(device, enabled):
    return torch.autocast(device_type=device.type, dtype=amp_dtype() if device.type == "cuda" else torch.bfloat16,
                          enabled=enabled and device.type == "cuda")


def find_batch_size(model, cfg, device, candidates=(96, 64, 48, 32, 24, 16, 12, 8, 6, 4, 2), budget=0.85):
    """Largest micro-batch whose forward+backward fits in `budget` of the currently free VRAM."""
    if device.type != "cuda":
        return int(cfg["batch_size"]) if isinstance(cfg["batch_size"], int) else 4
    S = cfg["img_size"]
    eff = int(cfg.get("eff_batch", 32))
    candidates = [c for c in candidates if eff % c == 0] or [2]
    saved = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}  # BN stats stay clean
    model.train()
    torch.cuda.empty_cache()
    free, _ = torch.cuda.mem_get_info()
    base = torch.cuda.memory_allocated()
    params = sum(p.numel() for p in model.parameters())
    optim_bytes = params * 4 * 3  # grads + AdamW moments
    for bs in [c for c in candidates if c <= cfg.get("max_batch", 64)]:
        ok = False
        x = out = loss = None
        try:
            torch.cuda.reset_peak_memory_stats()
            x = torch.randint(0, 255, (bs, 3, S, S), dtype=torch.uint8, device=device)
            if cfg.get("channels_last", True):
                x = x.contiguous(memory_format=torch.channels_last)
            with _autocast(device, cfg.get("amp", True)):
                out = model(x)
            loss = out["logits"].float().mean()
            if "seg" in out:
                loss = loss + out["seg"].float().mean()
            loss.backward()
            peak = torch.cuda.max_memory_allocated() - base + optim_bytes
            ok = peak < budget * free
        except torch.cuda.OutOfMemoryError:
            ok = False
        finally:
            model.zero_grad(set_to_none=True)
            x = out = loss = None
            gc.collect()
            torch.cuda.empty_cache()
        if ok:
            model.load_state_dict(saved)
            return bs
    model.load_state_dict(saved)
    return 2


def _param_groups(params, lr, wd):
    decay = [p for p in params if p.requires_grad and p.ndim > 1]
    no_decay = [p for p in params if p.requires_grad and p.ndim <= 1]
    groups = []
    if decay:
        groups.append(dict(params=decay, lr=lr, weight_decay=wd, initial_lr=lr))
    if no_decay:
        groups.append(dict(params=no_decay, lr=lr, weight_decay=0.0, initial_lr=lr))
    return groups


def _cosine(step, total, warm, floor=0.01):
    if step < warm:
        return (step + 1) / max(1, warm)
    t = (step - warm) / max(1, total - warm)
    return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(1.0, t)))


@torch.no_grad()
def predict(model, loader, device, amp=True, tta_flip=False, want_seg=True, cam_peaks=False, channels_last=True,
            progress=False):
    model.eval()
    probs, targets, segs, peaks, idxs = [], [], [], [], []
    it = loader
    if progress:
        try:
            from tqdm.auto import tqdm
            it = tqdm(loader, desc="predict", leave=False)
        except Exception:
            pass
    grid = None
    for x, _, y, idx in it:
        x = x.to(device, non_blocking=True)
        if channels_last and device.type == "cuda":
            x = x.contiguous(memory_format=torch.channels_last)
        with _autocast(device, amp):
            out = model(x)
            if tta_flip:
                out2 = model(torch.flip(x, dims=[-1]))
        logits = out["logits"].float()
        seg = out.get("seg")
        if tta_flip:
            logits = 0.5 * (logits + out2["logits"].float())
            if seg is not None:
                seg = 0.5 * (seg.float() + torch.flip(out2["seg"].float(), dims=[-1]))
        probs.append(torch.sigmoid(logits).cpu().numpy())
        targets.append(y.numpy())
        idxs.append(idx.numpy())
        if want_seg and seg is not None:
            segs.append((torch.sigmoid(seg.float())[:, 0] * 255).round().to(torch.uint8).cpu().numpy())
        if cam_peaks and "cams" in out:
            cams = out["cams"].float()
            b, c, h, w = cams.shape
            grid = h
            flat = cams.flatten(2).argmax(-1)
            peaks.append(torch.stack([flat // w, flat % w], -1).to(torch.int16).cpu().numpy())
    res = dict(probs=np.concatenate(probs), targets=np.concatenate(targets), idx=np.concatenate(idxs))
    res["seg"] = np.concatenate(segs) if segs else None
    res["cam_peaks"] = np.concatenate(peaks) if peaks else None
    res["cam_grid"] = grid
    return res


def load_weights(model, path, device):
    sd = torch.load(path, map_location=device, weights_only=True)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"[load_weights] missing={len(missing)} unexpected={len(unexpected)}")
    return model


def train_model(cfg, train_ds, val_ds, run_dir, pos_counts=None, log=print, keep_last=False):
    """Train one configuration. Resumable: re-running continues from last.pt; a finished run is skipped."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    done = run_dir / "done.json"
    if done.exists():
        summ = load_json(done)
        log(f"[{run_dir.name}] already trained (best val mAP {summ.get('best_val_map', float('nan')):.4f}) - skipping")
        return summ
    save_json(cfg, run_dir / "config.json")
    set_seed(cfg.get("seed", 0))
    device = get_device()
    model = build_model(cfg).to(device)
    cl = bool(cfg.get("channels_last", True)) and device.type == "cuda"
    if cl:
        model = model.to(memory_format=torch.channels_last)
    bs = cfg["batch_size"] if isinstance(cfg["batch_size"], int) else find_batch_size(model, cfg, device)
    accum = max(1, int(round(cfg["eff_batch"] / bs)))
    nw = cfg.get("num_workers", 4)
    train_loader = make_loader(train_ds, bs, True, nw, drop_last=len(train_ds) > bs)
    val_loader = make_loader(val_ds, bs * 2, False, nw)
    bb, new = model.new_param_groups()
    groups = _param_groups(bb, cfg["backbone_lr"], cfg["weight_decay"]) + \
        _param_groups(new, cfg["lr"], cfg["weight_decay"])
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.999))
    epochs = int(cfg["epochs"])
    steps_per_epoch = max(1, len(train_loader) // accum)
    total_steps = steps_per_epoch * epochs
    warm = int(cfg.get("warmup_epochs", 1.0) * steps_per_epoch)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: _cosine(s, total_steps, warm))
    use_amp = bool(cfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype() == torch.float16)
    crit = build_cls_loss(cfg, pos_counts).to(device)
    lam_seg = float(cfg.get("lambda_seg", 0.0)) if cfg.get("seg", False) else 0.0

    start_epoch, best, history, bad_epochs = 0, -1.0, [], 0
    last = run_dir / "last.pt"
    if last.exists():
        try:
            ck = torch.load(last, map_location=device, weights_only=False)
            model.load_state_dict(ck["model"])
            opt.load_state_dict(ck["opt"])
            sched.load_state_dict(ck["sched"])
            scaler.load_state_dict(ck["scaler"])
            start_epoch, best, history = ck["epoch"] + 1, ck["best"], ck["history"]
            bad_epochs = ck.get("bad_epochs", 0)
            log(f"[{run_dir.name}] resuming after epoch {start_epoch} of {cfg['epochs']}")
            del ck
        except Exception as e:  # noqa: BLE001
            log(f"[{run_dir.name}] could not read last.pt ({e}); starting this run from scratch")
            start_epoch, best, history, bad_epochs = 0, -1.0, [], 0
    log(f"[{run_dir.name}] params={count_params(model):.2f}M micro-batch={bs} x accum {accum} "
        f"| {len(train_ds)} train / {len(val_ds)} val | {epochs} epochs | device={device}")
    t_start = time.time()
    seen = 0
    for epoch in range(start_epoch, epochs):
        model.train()
        t0 = time.time()
        acc = torch.zeros(3, device=device)
        n = 0
        opt.zero_grad(set_to_none=True)
        it = train_loader
        try:
            from tqdm.auto import tqdm
            it = tqdm(train_loader, desc=f"{run_dir.name} ep{epoch + 1}/{epochs}", leave=False)
        except Exception:
            pass
        for k, (x, m, y, _) in enumerate(it):
            x = x.to(device, non_blocking=True)
            if cl:
                x = x.contiguous(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            with _autocast(device, use_amp):
                out = model(x)
            lc = crit(out["logits"], y)
            loss = lc
            ls = torch.zeros((), device=device)
            if lam_seg > 0 and "seg" in out:
                ls = seg_loss(out["seg"], m.to(device, non_blocking=True).float() / 255.0)
                loss = loss + lam_seg * ls
            if not torch.isfinite(loss):
                log(f"non-finite loss at epoch {epoch + 1} step {k}; skipping batch")
                opt.zero_grad(set_to_none=True)
                continue
            scaler.scale(loss / accum).backward()
            if (k + 1) % accum == 0:
                if cfg.get("grad_clip"):
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
            acc += torch.stack([loss.detach().float(), lc.detach().float(), ls.detach().float()])
            n += 1
            seen += len(x)
            if hasattr(it, "set_postfix") and k % 50 == 0:
                it.set_postfix(loss=f"{float(acc[0]) / max(n, 1):.3f}")
        cuda_sync()
        t_train = time.time() - t0
        run_loss, run_cls, run_seg = [float(v) for v in acc.cpu()]
        row = dict(epoch=epoch + 1, loss=run_loss / max(n, 1), cls_loss=run_cls / max(n, 1),
                   seg_loss=run_seg / max(n, 1), lr=opt.param_groups[-1]["lr"], train_time=t_train,
                   img_per_s=n * bs / max(t_train, 1e-6))
        if (epoch + 1) % cfg.get("eval_every", 1) == 0 or epoch + 1 == epochs:
            pr = predict(model, val_loader, device, use_amp, want_seg=False, channels_last=cl)
            vmap = mean_ap(pr["targets"], pr["probs"])
            row["val_mAP"] = vmap
            if hasattr(model, "gate_values"):
                row.update(model.gate_values())
            if vmap > best:
                best = vmap
                bad_epochs = 0
                torch_save_atomic(model.state_dict(), run_dir / "best.pt")
            else:
                bad_epochs += 1
        history.append(row)
        eta = (time.time() - t_start) / (epoch + 1 - start_epoch) * (epochs - epoch - 1)
        log(f"[{run_dir.name}] ep {epoch + 1}/{epochs} loss {row['loss']:.4f} (cls {row['cls_loss']:.4f} "
            f"seg {row['seg_loss']:.4f}) val mAP {row.get('val_mAP', float('nan')):.4f} best {best:.4f} "
            f"| {row['img_per_s']:.0f} img/s | ETA {fmt_time(eta)}")
        torch_save_atomic(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                               scaler=scaler.state_dict(), epoch=epoch, best=best, history=history,
                               bad_epochs=bad_epochs), last)
        save_json(dict(epoch=epoch + 1, epochs=epochs, best_val_map=best,
                       minutes=sum(h.get("train_time", 0) for h in history) / 60), run_dir / "progress.json")
        if cfg.get("patience") and bad_epochs >= cfg["patience"]:
            log(f"[{run_dir.name}] early stop after {epoch + 1} epochs")
            break
    if not (run_dir / "best.pt").exists():
        torch_save_atomic(model.state_dict(), run_dir / "best.pt")
    summ = dict(best_val_map=best, epochs_run=len(history), micro_batch=bs, accum=accum,
                params_m=count_params(model), train_minutes=sum(h.get("train_time", 0) for h in history) / 60,
                peak_vram_gb=(torch.cuda.max_memory_allocated() / 1e9) if device.type == "cuda" else 0.0,
                history=history)
    save_json(summ, done)
    if not keep_last and last.exists():
        os.remove(last)
    del model, opt
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summ


def load_run_model(cfg, run_dir, device=None):
    device = device or get_device()
    model = build_model(dict(cfg, pretrained=False)).to(device)
    load_weights(model, Path(run_dir) / "best.pt", device)
    if device.type == "cuda" and cfg.get("channels_last", True):
        model = model.to(memory_format=torch.channels_last)
    return model.eval()


@torch.no_grad()
def measure_latency(model, img_size, device, n=30, bs=1, warmup=5, amp=True):
    model.eval()
    x = torch.randint(0, 255, (bs, 3, img_size, img_size), dtype=torch.uint8, device=device)
    if device.type == "cuda":
        x = x.contiguous(memory_format=torch.channels_last)
    for _ in range(warmup):
        with _autocast(device, amp):
            model(x)
    cuda_sync()
    t = time.time()
    for _ in range(n):
        with _autocast(device, amp):
            model(x)
    cuda_sync()
    return (time.time() - t) / n / bs * 1000.0


__all__ = ["make_loader", "train_model", "predict", "load_run_model", "measure_latency", "find_batch_size",
           "load_weights"]
