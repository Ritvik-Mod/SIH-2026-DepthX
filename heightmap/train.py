"""Training loop.

Three parameter groups because the three parts of the network know different amounts;
warmup because at step 0 the new head is random and its gradients are noise that would
damage the pretrained encoder; bf16 rather than fp16 because SILog takes logs and square
roots and fp16's narrow range turns those into NaN.
"""
from __future__ import annotations
import argparse, json, math, os, time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from . import config
from .data.gamus import GamusDataset, ConcatSplits, build_splits
from .models.net import HeightNet
from .losses.combined import CombinedLoss
from .metrics import MetricAccumulator, format_table, sharpness
from .utils.misc import seed_everything, pick_device, amp_dtype_for, JsonlLogger, EMA, warmup_cosine
from .utils.viz import panel

# DataLoader workers hand tensors to the main process as file descriptors under the
# default 'file_descriptor' sharing strategy, and the main process's fd budget is
# finite.  With 14 workers over a 6,557-tile split that ceiling was hit at epoch 23,
# identically in two separate runs -- OSError: [Errno 24] Too many open files.  The
# 'file_system' strategy passes shared-memory names instead and does not consume an fd
# per tensor.  The official-split runs survived only because the ladder ran one job at
# a time; three concurrent jobs is what exposed it.
torch.multiprocessing.set_sharing_strategy("file_system")


def make_loaders(cfg):
    tr, va, te = build_splits(cfg)
    train_ds = ConcatSplits(cfg, tr, train=True)
    val_ds = ConcatSplits(cfg, va, train=False)
    nw = int(cfg.data.num_workers)
    common = dict(num_workers=nw, pin_memory=False, persistent_workers=nw > 0)
    return (DataLoader(train_ds, batch_size=int(cfg.train.batch_size), shuffle=True,
                       drop_last=True, **common),
            DataLoader(val_ds, batch_size=int(cfg.train.batch_size), shuffle=False, **common),
            te)


@torch.no_grad()
def validate(model, loader, device, cfg, limit_batches=None, save_panel=None):
    model.eval()
    acc = MetricAccumulator()
    sharp = []
    first = None
    for i, b in enumerate(loader):
        if limit_batches and i >= limit_batches:
            break
        x = b["image"].to(device); gsd = b["gsd"].to(device)
        out = model(x, gsd)
        p = out["height"].float().cpu()
        # Building-only metrics must use the GROUND-TRUTH class mask, never the model's
        # own predicted semantics: a model that predicts no buildings would otherwise
        # score no building error, and the metric is undefined early in training when
        # the semantic head still predicts a single class everywhere.
        acc.update(p, b["height"], b["mask"], b["cls"])
        sharp.append(sharpness(p, b["height"], b["mask"])["ratio"])
        if first is None:
            first = (b, p)
    res = acc.result()
    s = [v for v in sharp if np.isfinite(v)]
    res["sharpness_ratio"] = float(np.mean(s)) if s else float("nan")
    if save_panel and first is not None:
        b, p = first
        from .data.transforms import IMAGENET_MEAN, IMAGENET_STD
        img = (b["image"][0].permute(1, 2, 0).numpy() * IMAGENET_STD + IMAGENET_MEAN).clip(0, 1)
        gt = b["height"][0, 0].numpy(); pr = p[0, 0].numpy()
        try:
            import imageio.v2 as imageio
            imageio.imwrite(save_panel, panel(img, pr, gt, pr - gt))
        except Exception:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            plt.imsave(save_panel, panel(img, pr, gt, pr - gt))
    model.train()
    return res


def main(argv=None):
    cfg = config.load_from_argv(argv)
    seed_everything(int(cfg.seed))
    out = Path(cfg.out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(json.dumps(json.loads(json.dumps(
        __import__("omegaconf").OmegaConf.to_container(cfg, resolve=True))), indent=2))
    log = JsonlLogger(out / "log.jsonl")

    device = pick_device()
    amp_dt = amp_dtype_for(device, str(cfg.train.amp_dtype))
    print(f"device={device}  amp={amp_dt}  out={out}")

    train_dl, val_dl, _ = make_loaders(cfg)
    model = HeightNet(cfg).to(device)
    lossfn = CombinedLoss(cfg)

    # Block surgery BEFORE param_groups: a frozen block must never receive an optimiser
    # entry, or it would accumulate gradients that nothing applies -- trainable by every
    # assertion and silently frozen in fact.
    spec = str(getattr(cfg.train, "trainable_blocks", "all"))
    kept = model.set_trainable_blocks(spec, bool(getattr(cfg.train, "train_embed", False)))
    if not bool(getattr(cfg.train, "train_neck", True)):
        model.set_neck_trainable(False)
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable blocks [{spec}] -> {len(kept)}/{model.n_blocks()} "
          f"{'' if len(kept) > 8 else kept}  neck={cfg.train.train_neck}  "
          f"embed={cfg.train.train_embed}  llrd={cfg.train.llrd}  "
          f"trainable params {n_tr/1e6:.1f}M")

    groups = model.param_groups(float(cfg.train.lr_encoder), float(cfg.train.lr_decoder),
                                float(cfg.train.lr_head), float(cfg.train.weight_decay),
                                float(getattr(cfg.train, "llrd", 1.0)))
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.999), eps=1e-8)
    base_lrs = [g["lr"] for g in opt.param_groups]
    scaler = torch.amp.GradScaler(enabled=(amp_dt == torch.float16))
    ema = EMA(model, float(cfg.train.ema_decay)) if cfg.train.ema_decay else None

    accum = max(1, int(cfg.train.accum_steps))
    steps_per_epoch = max(1, len(train_dl) // accum)
    total_steps = steps_per_epoch * int(cfg.train.epochs)
    warm = int(cfg.train.warmup_steps)
    print(f"train tiles={len(train_dl.dataset)} val={len(val_dl.dataset)} "
          f"steps/epoch={steps_per_epoch} total={total_steps}")

    # ---- resume ---------------------------------------------------------------
    # Shared clusters kill jobs at the walltime limit, so a run MUST be able to pick up
    # where it stopped.  Everything that carries state is saved and restored: model,
    # optimiser moments, grad scaler, EMA shadow, epoch, step and the best score.
    step, best, start_epoch = 0, float("inf"), 0
    resume = str(getattr(cfg.train, "resume", "auto"))
    ck_path = out / "last.pt" if resume == "auto" else (None if resume == "never" else Path(resume))
    if ck_path is not None and ck_path.exists():
        ck = torch.load(ck_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        if "optim" in ck:
            opt.load_state_dict(ck["optim"])
        if "scaler" in ck and ck["scaler"] is not None:
            scaler.load_state_dict(ck["scaler"])
        if ema is not None and "ema" in ck:
            ema.load_state_dict(ck["ema"])
        start_epoch = int(ck.get("epoch", -1)) + 1
        step = int(ck.get("step", 0))
        best = float(ck.get("best", float("inf")))
        print(f"resumed from {ck_path} at epoch {start_epoch}, step {step}, best {best:.4f}")
    elif ck_path is not None:
        print(f"no checkpoint at {ck_path}; starting fresh")

    # ---- warm start from ANOTHER run's weights --------------------------------
    # Distinct from resume.  resume continues THIS run and restores epoch/optimiser, so
    # pointing it at a finished run's checkpoint makes start_epoch land past the epoch
    # count and the job exits having done nothing.  init_from takes only the weights and
    # starts at epoch 0 with a fresh optimiser, which is what "fine-tune from A5" means.
    # It is skipped when a resume already happened, so a walltime kill still continues.
    init_from = str(getattr(cfg.train, "init_from", "") or "")
    if init_from and start_epoch == 0:
        ip = Path(init_from)
        if not ip.exists():
            raise FileNotFoundError(
                f"train.init_from={ip} does not exist. Refusing to silently train from "
                f"the pretrained backbone instead: that is a different experiment.")
        ick = torch.load(ip, map_location=device, weights_only=False)
        sd = ick.get("model", ick)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"init_from {ip}: loaded weights, epoch/optimiser left fresh "
              f"({len(missing)} missing, {len(unexpected)} unexpected keys)")
        if missing or unexpected:
            print(f"  missing={list(missing)[:8]}\n  unexpected={list(unexpected)[:8]}")

    if start_epoch >= int(cfg.train.epochs):
        print(f"already finished ({start_epoch}/{cfg.train.epochs} epochs). nothing to do.")
        return

    for epoch in range(start_epoch, int(cfg.train.epochs)):
        # The encoder is frozen at first so the random head can become sane before its
        # noisy gradients reach 142M-image pretrained features.
        model.set_encoder_trainable(epoch >= int(cfg.train.freeze_encoder_epochs))
        model.train()
        pbar = tqdm(train_dl, desc=f"epoch {epoch}", dynamic_ncols=True)
        opt.zero_grad(set_to_none=True)
        for it, b in enumerate(pbar):
            for g, bl in zip(opt.param_groups, base_lrs):
                g["lr"] = warmup_cosine(step, warm, total_steps, bl, float(cfg.train.min_lr))
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
            ctx = torch.autocast(device_type=device.type, dtype=amp_dt) if amp_dt else \
                torch.autocast(device_type=device.type, enabled=False)
            with ctx:
                out_d = model(batch["image"], batch["gsd"])
                loss, logs = lossfn(out_d, batch)
            scaler.scale(loss / accum).backward()

            if (it + 1) % accum == 0:
                if cfg.train.clip_grad:
                    scaler.unscale_(opt)
                    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.train.clip_grad))
                    logs["grad_norm"] = float(gn)
                scaler.step(opt); scaler.update()
                opt.zero_grad(set_to_none=True)
                if ema: ema.update(model)
                step += 1
                if step % int(cfg.train.log_every) == 0:
                    logs.update(step=step, epoch=epoch, lr_head=opt.param_groups[-1]["lr"])
                    log.log(**logs)
                    pbar.set_postfix({k.replace("loss/", ""): round(v, 3)
                                      for k, v in logs.items() if k.startswith("loss/")})

        if (epoch + 1) % int(cfg.train.val_every) == 0:
            res = validate(model, val_dl, device, cfg, save_panel=str(out / f"panel_e{epoch}.png"))
            bm = res["building"]["mae"]
            print(f"\n[epoch {epoch}] val\n{format_table(res)}\n"
                  f"sharpness ratio {res['sharpness_ratio']:.3f}")
            log.log(step=step, epoch=epoch, phase="val",
                    **{f"val/{k}": v for k, v in res["overall"].items()},
                    **{f"valb/{k}": v for k, v in res["building"].items()},
                    sharpness=res["sharpness_ratio"])
            ck = {"model": model.state_dict(), "cfg": __import__("omegaconf").OmegaConf.to_container(cfg),
                  "epoch": epoch, "step": step, "val": res,
                  "optim": opt.state_dict(),
                  "scaler": scaler.state_dict() if scaler.is_enabled() else None,
                  "best": min(best, bm) if np.isfinite(bm) else best}
            if ema: ck["ema"] = ema.state_dict()
            tmp = out / "last.pt.tmp"
            torch.save(ck, tmp)
            os.replace(tmp, out / "last.pt")   # atomic: a kill mid-write must not corrupt it
            # checkpoint on building MAE, not total loss: the loss mixes six terms and
            # a drop in it does not necessarily mean better heights.
            if np.isfinite(bm) and bm < best:
                best = bm
                torch.save(ck, out / "best.pt")
                print(f"  new best building MAE {bm:.3f} -> best.pt")
    print("done. best building MAE:", best)


if __name__ == "__main__":
    main()
