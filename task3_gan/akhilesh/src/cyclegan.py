#!/usr/bin/env python
"""
DATA 266 Lab 1, Task 3: CycleGAN, Monet (domain A) <-> Photo (domain B), trained from scratch.

G_AB: Monet -> Photo      G_BA: Photo -> Monet (the Kaggle direction)
D_A : real vs fake Monet  D_B : real vs fake Photo

Run from your member folder (task3_gan/<your_name>/):
    python src/cyclegan.py --config src/config.yaml --data_dir ../data --out_dir .
Resume (GPU lab sessions end, machines get wiped):
    python src/cyclegan.py --config src/config.yaml --data_dir ../data --out_dir . --resume checkpoints/last_full.pt
"""
import argparse
import csv
import glob
import json
import logging
import math
import os
import platform
import random
import subprocess
import sys
import time
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset


# ----------------------------------------------------------------------------
# config, logging, manifest
# ----------------------------------------------------------------------------
def load_cfg(path):
    with open(path) as f:
        if path.endswith((".yaml", ".yml")):
            import yaml
            return yaml.safe_load(f) or {}
        return json.load(f)


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    p.add_argument("--data_dir", default="../data", help="folder containing monet_jpg/ and photo_jpg/")
    p.add_argument("--out_dir", default=".")
    p.add_argument("--run_name", default="cyclegan")
    p.add_argument("--seed", type=int, default=266)
    # architecture
    p.add_argument("--ngf", type=int, default=64)
    p.add_argument("--ndf", type=int, default=64)
    p.add_argument("--g_blocks", type=int, default=9)
    p.add_argument("--d_layers", type=int, default=3)
    p.add_argument("--upsample", default="deconv", choices=["deconv", "resize"])
    p.add_argument("--spectral_norm", type=int, default=0)
    p.add_argument("--g_dropout", type=float, default=0.0)
    # losses
    p.add_argument("--gan_loss", default="lsgan", choices=["lsgan", "vanilla"])
    p.add_argument("--lambda_cyc", type=float, default=10.0)
    p.add_argument("--lambda_id", type=float, default=0.5, help="identity weight, as a fraction of lambda_cyc")
    p.add_argument("--pool_size", type=int, default=50)
    # optimisation
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lr_d_mult", type=float, default=1.0)
    p.add_argument("--beta1", type=float, default=0.5)
    p.add_argument("--n_epochs", type=int, default=15, help="epochs at constant lr")
    p.add_argument("--n_epochs_decay", type=int, default=15, help="epochs of linear decay to 0")
    p.add_argument("--steps_per_epoch", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--load_size", type=int, default=286)
    p.add_argument("--crop_size", type=int, default=256)
    p.add_argument("--no_flip", type=int, default=0)
    p.add_argument("--amp", type=int, default=0, help="bf16 autocast; off by default for GAN stability")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--save_every", type=int, default=5)
    p.add_argument("--resume", default=None)
    pre, _ = p.parse_known_args(argv)
    if pre.config:
        p.set_defaults(**load_cfg(pre.config))
    return p.parse_args(argv)


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def device_name():
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return platform.processor() or "CPU"


def setup_run(args, tag="train"):
    for d in ("checkpoints", "outputs", "logs"):
        os.makedirs(os.path.join(args.out_dir, d), exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(args.out_dir, "logs", f"{args.run_name}_{tag}_{stamp}.log")
    logger = logging.getLogger(f"{args.run_name}_{tag}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(message)s")
    for h in (logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)):
        h.setFormatter(fmt); logger.addHandler(h)
    manifest = {"run_name": args.run_name, "tag": tag, "timestamp": stamp, "args": vars(args),
                "python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
                "cuda": torch.version.cuda, "device": device_name()}
    try:
        manifest["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        manifest["git_commit"] = None
    with open(os.path.join(args.out_dir, "logs", f"manifest_{args.run_name}_{tag}_{stamp}.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    try:
        freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"]).decode()
        with open(os.path.join(args.out_dir, "logs", f"pip_freeze_{stamp}.txt"), "w") as f:
            f.write(freeze)
    except Exception:
        pass
    logger.info(f"log file: {log_path} | device: {device_name()}")
    logger.info("args: " + json.dumps(vars(args)))
    return logger


# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------
def list_images(folder):
    # glob is case-insensitive on Windows, so "*.jpg" and "*.JPG" match the same files; de-duplicate
    seen, files = set(), []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG"):
        for f in glob.glob(os.path.join(folder, ext)):
            key = os.path.normcase(os.path.abspath(f))
            if key not in seen:
                seen.add(key)
                files.append(f)
    return sorted(files)


class UnpairedImages(Dataset):
    def __init__(self, folder, load_size, crop_size, flip):
        import torchvision.transforms as T
        self.files = list_images(folder)
        if not self.files:
            raise FileNotFoundError(f"no images in {folder}")
        tf = [T.Resize((load_size, load_size), interpolation=T.InterpolationMode.BICUBIC),
              T.RandomCrop(crop_size)]
        if flip:
            tf.append(T.RandomHorizontalFlip())
        tf += [T.ToTensor(), T.Normalize([0.5] * 3, [0.5] * 3)]
        self.tf = T.Compose(tf)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        return self.tf(Image.open(self.files[i]).convert("RGB"))


def infinite(loader):
    while True:
        for b in loader:
            yield b


# ----------------------------------------------------------------------------
# networks
# ----------------------------------------------------------------------------
class ResBlock(nn.Module):
    def __init__(self, c, dropout=0.0):
        super().__init__()
        layers = [nn.ReflectionPad2d(1), nn.Conv2d(c, c, 3), nn.InstanceNorm2d(c), nn.ReLU(True)]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers += [nn.ReflectionPad2d(1), nn.Conv2d(c, c, 3), nn.InstanceNorm2d(c)]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.block(x)


class ResnetGenerator(nn.Module):
    def __init__(self, ngf=64, n_blocks=9, upsample="deconv", dropout=0.0):
        super().__init__()
        m = [nn.ReflectionPad2d(3), nn.Conv2d(3, ngf, 7), nn.InstanceNorm2d(ngf), nn.ReLU(True)]
        ch = ngf
        for _ in range(2):
            m += [nn.Conv2d(ch, ch * 2, 3, stride=2, padding=1), nn.InstanceNorm2d(ch * 2), nn.ReLU(True)]
            ch *= 2
        m += [ResBlock(ch, dropout) for _ in range(n_blocks)]
        for _ in range(2):
            if upsample == "deconv":
                m += [nn.ConvTranspose2d(ch, ch // 2, 3, stride=2, padding=1, output_padding=1)]
            else:  # resize-conv, avoids checkerboard artifacts
                m += [nn.Upsample(scale_factor=2, mode="nearest"), nn.ReflectionPad2d(1), nn.Conv2d(ch, ch // 2, 3)]
            m += [nn.InstanceNorm2d(ch // 2), nn.ReLU(True)]
            ch //= 2
        m += [nn.ReflectionPad2d(3), nn.Conv2d(ch, 3, 7), nn.Tanh()]
        self.net = nn.Sequential(*m)

    def forward(self, x):
        return self.net(x)


class PatchDiscriminator(nn.Module):
    """70x70 PatchGAN when n_layers=3."""

    def __init__(self, ndf=64, n_layers=3, sn=False):
        super().__init__()
        wrap = nn.utils.spectral_norm if sn else (lambda x: x)
        m = [wrap(nn.Conv2d(3, ndf, 4, 2, 1)), nn.LeakyReLU(0.2, True)]
        mult = 1
        for i in range(1, n_layers):
            prev, mult = mult, min(2 ** i, 8)
            m += [wrap(nn.Conv2d(ndf * prev, ndf * mult, 4, 2, 1)), nn.InstanceNorm2d(ndf * mult),
                  nn.LeakyReLU(0.2, True)]
        prev, mult = mult, min(2 ** n_layers, 8)
        m += [wrap(nn.Conv2d(ndf * prev, ndf * mult, 4, 1, 1)), nn.InstanceNorm2d(ndf * mult),
              nn.LeakyReLU(0.2, True), wrap(nn.Conv2d(ndf * mult, 1, 4, 1, 1))]
        self.net = nn.Sequential(*m)

    def forward(self, x):
        return self.net(x)


def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        w = m.weight_orig if hasattr(m, "weight_orig") else m.weight
        nn.init.normal_(w, 0.0, 0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def build_generator(a):
    a = a if isinstance(a, dict) else vars(a)
    return ResnetGenerator(a["ngf"], a["g_blocks"], a["upsample"], a.get("g_dropout", 0.0))


def build_discriminator(a):
    a = a if isinstance(a, dict) else vars(a)
    return PatchDiscriminator(a["ndf"], a["d_layers"], bool(a["spectral_norm"]))


def n_params(m):
    return sum(p.numel() for p in m.parameters())


class ImagePool:
    """History buffer of generated images (Shrivastava et al.), as in the CycleGAN paper."""

    def __init__(self, size):
        self.size, self.images = size, []

    def query(self, imgs):
        if self.size == 0:
            return imgs
        out = []
        for img in imgs:
            img = img.unsqueeze(0)
            if len(self.images) < self.size:
                self.images.append(img); out.append(img)
            elif random.random() > 0.5:
                i = random.randint(0, self.size - 1)
                out.append(self.images[i].clone()); self.images[i] = img
            else:
                out.append(img)
        return torch.cat(out, 0)


def gan_loss(pred, real, kind):
    pred = pred.float()
    target = torch.ones_like(pred) if real else torch.zeros_like(pred)
    if kind == "lsgan":
        return F.mse_loss(pred, target)
    return F.binary_cross_entropy_with_logits(pred, target)


def set_requires_grad(nets, flag):
    for n in nets:
        for p in n.parameters():
            p.requires_grad_(flag)


def grad_norm(params):
    norms = [p.grad.detach().float().norm() for p in params if p.grad is not None]
    return torch.norm(torch.stack(norms)).item() if norms else 0.0


# ----------------------------------------------------------------------------
# plotting (also used by evaluate_local.py)
# ----------------------------------------------------------------------------
LOG_COLS = ["step", "epoch", "lr", "G_adv_AB", "G_adv_BA", "cyc_A", "cyc_B", "idt_A", "idt_B",
            "loss_G", "D_A", "D_B", "gnorm_G", "gnorm_D", "nonfinite"]


def read_step_log(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return {c: np.array([float(r[c]) for r in rows]) for c in LOG_COLS}


def plot_training(csv_path, out_png, k=200):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = read_step_log(csv_path)
    s = d["step"]

    def sm(v):
        if len(v) <= k:
            return s, v
        return s[k - 1:], np.convolve(v, np.ones(k) / k, mode="valid")
    fig, ax = plt.subplots(2, 2, figsize=(14, 9))
    for c in ("G_adv_AB", "G_adv_BA", "D_A", "D_B"):
        ax[0, 0].plot(*sm(d[c]), label=c)
    ax[0, 0].set_title(f"Adversarial losses ({k}-step avg)"); ax[0, 0].legend()
    for c in ("cyc_A", "cyc_B", "idt_A", "idt_B"):
        ax[0, 1].plot(*sm(d[c]), label=c)
    ax[0, 1].set_title("Cycle-consistency and identity losses (weighted)"); ax[0, 1].legend()
    ax[1, 0].plot(s, d["gnorm_G"], lw=0.3, label="G"); ax[1, 0].plot(s, d["gnorm_D"], lw=0.3, label="D")
    ax[1, 0].set_yscale("log"); ax[1, 0].set_title("Gradient norms"); ax[1, 0].legend()
    ax[1, 1].plot(s, d["lr"]); ax[1, 1].set_title("Learning rate")
    for a in ax.flat:
        a.set_xlabel("step")
    fig.tight_layout(); fig.savefig(out_png, dpi=150); plt.close(fig)


# ----------------------------------------------------------------------------
# training
# ----------------------------------------------------------------------------
def main():
    args = parse_args()
    set_seed(args.seed)
    log = setup_run(args, "train")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = bool(args.amp) and device == "cuda" and torch.cuda.is_bf16_supported()
    ac = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if use_amp else nullcontext

    dsA = UnpairedImages(os.path.join(args.data_dir, "monet_jpg"), args.load_size, args.crop_size, not args.no_flip)
    dsB = UnpairedImages(os.path.join(args.data_dir, "photo_jpg"), args.load_size, args.crop_size, not args.no_flip)
    log.info(f"domain A (Monet): {len(dsA)} images | domain B (Photo): {len(dsB)} images")
    mk = lambda ds: DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                               pin_memory=device == "cuda", drop_last=True, persistent_workers=args.num_workers > 0)
    itA, itB = infinite(mk(dsA)), infinite(mk(dsB))

    G_AB, G_BA = build_generator(args).to(device), build_generator(args).to(device)
    D_A, D_B = build_discriminator(args).to(device), build_discriminator(args).to(device)
    for n in (G_AB, G_BA, D_A, D_B):
        n.apply(init_weights)
    pc = {"G_AB": n_params(G_AB), "G_BA": n_params(G_BA), "D_A": n_params(D_A), "D_B": n_params(D_B)}
    pc["total"] = sum(pc.values())
    log.info("parameter counts: " + json.dumps(pc))

    g_params = list(G_AB.parameters()) + list(G_BA.parameters())
    d_params = list(D_A.parameters()) + list(D_B.parameters())
    opt_G = torch.optim.Adam(g_params, lr=args.lr, betas=(args.beta1, 0.999))
    opt_D = torch.optim.Adam(d_params, lr=args.lr * args.lr_d_mult, betas=(args.beta1, 0.999))
    total_epochs = args.n_epochs + args.n_epochs_decay
    rule = lambda e: 1.0 - max(0, e - args.n_epochs + 1) / float(args.n_epochs_decay + 1)
    sch_G = torch.optim.lr_scheduler.LambdaLR(opt_G, rule)
    sch_D = torch.optim.lr_scheduler.LambdaLR(opt_D, rule)
    pool_A, pool_B = ImagePool(args.pool_size), ImagePool(args.pool_size)

    start_epoch, step, train_time, nonfinite_total = 1, 0, 0.0, 0
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        arch = ("ngf", "ndf", "g_blocks", "d_layers", "upsample", "spectral_norm", "g_dropout")
        diff = {k: (ck["args"][k], getattr(args, k)) for k in arch if ck["args"].get(k) != getattr(args, k)}
        if diff:
            raise SystemExit(f"--resume: architecture args differ from the checkpoint (saved, now): {diff}")
        for name, net in (("G_AB", G_AB), ("G_BA", G_BA), ("D_A", D_A), ("D_B", D_B)):
            net.load_state_dict(ck[name])
        opt_G.load_state_dict(ck["opt_G"]); opt_D.load_state_dict(ck["opt_D"])
        sch_G.load_state_dict(ck["sch_G"]); sch_D.load_state_dict(ck["sch_D"])
        start_epoch, step = ck["epoch"] + 1, ck["step"]
        train_time, nonfinite_total = ck["train_time"], ck["nonfinite_total"]
        log.info(f"resumed from {args.resume}: epoch {ck['epoch']} step {step}")

    step_csv = os.path.join(args.out_dir, "logs", f"train_steps_{args.run_name}.csv")
    new = not os.path.exists(step_csv)
    fcsv = open(step_csv, "a", newline="")
    w = csv.writer(fcsv)
    if new:
        w.writerow(LOG_COLS)

    # fixed images for per-epoch sample grids
    fixed_A = torch.stack([dsA[i] for i in range(4)]).to(device)
    fixed_B = torch.stack([dsB[i] for i in range(4)]).to(device)
    sample_dir = os.path.join(args.out_dir, "outputs", "train_samples")
    os.makedirs(sample_dir, exist_ok=True)
    l1 = nn.L1Loss()
    lam_c, lam_i = args.lambda_cyc, args.lambda_id * args.lambda_cyc
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    log.info(f"training epochs {start_epoch}..{total_epochs}, {args.steps_per_epoch} steps each, bs={args.batch_size}")

    for epoch in range(start_epoch, total_epochs + 1):
        G_AB.train(); G_BA.train(); D_A.train(); D_B.train()
        t0 = time.time()
        acc = {c: 0.0 for c in LOG_COLS[3:14]}
        for it in range(args.steps_per_epoch):
            real_A = next(itA).to(device, non_blocking=True)
            real_B = next(itB).to(device, non_blocking=True)
            gG = gD = float("nan")

            # ---- generators ----
            set_requires_grad([D_A, D_B], False)
            opt_G.zero_grad(set_to_none=True)
            with ac():
                fake_B = G_AB(real_A); rec_A = G_BA(fake_B)
                fake_A = G_BA(real_B); rec_B = G_AB(fake_A)
                adv_AB = gan_loss(D_B(fake_B), True, args.gan_loss)
                adv_BA = gan_loss(D_A(fake_A), True, args.gan_loss)
                cyc_A = l1(rec_A.float(), real_A) * lam_c
                cyc_B = l1(rec_B.float(), real_B) * lam_c
                if lam_i > 0:
                    idt_A = l1(G_BA(real_A).float(), real_A) * lam_i
                    idt_B = l1(G_AB(real_B).float(), real_B) * lam_i
                else:
                    idt_A = idt_B = torch.zeros((), device=device)
                loss_G = adv_AB + adv_BA + cyc_A + cyc_B + idt_A + idt_B
            nonfinite = 0
            if not torch.isfinite(loss_G):
                nonfinite = 1
            else:
                loss_G.backward()
                gG = grad_norm(g_params)
                if math.isfinite(gG):
                    opt_G.step()
                else:
                    nonfinite = 1

            # ---- discriminators ----
            set_requires_grad([D_A, D_B], True)
            opt_D.zero_grad(set_to_none=True)
            with ac():
                fB = pool_B.query(fake_B.detach()); fA = pool_A.query(fake_A.detach())
                lD_B = 0.5 * (gan_loss(D_B(real_B), True, args.gan_loss) + gan_loss(D_B(fB), False, args.gan_loss))
                lD_A = 0.5 * (gan_loss(D_A(real_A), True, args.gan_loss) + gan_loss(D_A(fA), False, args.gan_loss))
                loss_D = lD_A + lD_B
            if not torch.isfinite(loss_D):
                nonfinite = 1
            else:
                loss_D.backward()
                gD = grad_norm(d_params)
                if math.isfinite(gD):
                    opt_D.step()
                else:
                    nonfinite = 1
            nonfinite_total += nonfinite

            vals = [adv_AB.item(), adv_BA.item(), cyc_A.item(), cyc_B.item(), idt_A.item(), idt_B.item(),
                    loss_G.item(), lD_A.item(), lD_B.item(), gG, gD, nonfinite]
            w.writerow([step, epoch, f"{opt_G.param_groups[0]['lr']:.3e}"] + [f"{v:.5f}" for v in vals[:-1]] + [nonfinite])
            for c, v in zip(LOG_COLS[3:14], vals[:-1]):
                acc[c] += v if math.isfinite(v) else 0.0
            if step % args.log_every == 0:
                el = time.time() - t0
                log.info(f"ep {epoch} it {it}/{args.steps_per_epoch} step {step} | G {vals[6]:.3f} "
                         f"(adv {vals[0]:.3f}/{vals[1]:.3f} cyc {vals[2]:.3f}/{vals[3]:.3f} idt {vals[4]:.3f}/{vals[5]:.3f}) "
                         f"| D {vals[7]:.3f}/{vals[8]:.3f} | gn {gG:.2f}/{gD:.2f} | {(it + 1) * args.batch_size / max(el, 1e-9):.1f} img/s")
                fcsv.flush()
            step += 1

        if device == "cuda":
            torch.cuda.synchronize()
        ep_time = time.time() - t0
        train_time += ep_time
        sch_G.step(); sch_D.step()
        log.info(f"EPOCH {epoch} done in {ep_time / 60:.1f} min | means: " +
                 json.dumps({c: round(v / args.steps_per_epoch, 4) for c, v in acc.items()}) +
                 f" | nonfinite so far {nonfinite_total}")
        fcsv.flush()

        # sample grid: rows = real, translated, reconstructed
        G_AB.eval(); G_BA.eval()
        with torch.no_grad():
            fb = G_AB(fixed_A); ra = G_BA(fb); fa = G_BA(fixed_B); rb = G_AB(fa)
        import torchvision.utils as vutils
        grid = torch.cat([fixed_A, fb, ra, fixed_B, fa, rb]) * 0.5 + 0.5
        vutils.save_image(grid, os.path.join(sample_dir, f"epoch_{epoch:03d}.jpg"), nrow=4)

        ck = {"G_AB": G_AB.state_dict(), "G_BA": G_BA.state_dict(), "D_A": D_A.state_dict(),
              "D_B": D_B.state_dict(), "opt_G": opt_G.state_dict(), "opt_D": opt_D.state_dict(),
              "sch_G": sch_G.state_dict(), "sch_D": sch_D.state_dict(), "epoch": epoch, "step": step,
              "train_time": train_time, "nonfinite_total": nonfinite_total, "args": vars(args)}
        torch.save(ck, os.path.join(args.out_dir, "checkpoints", "last_full.pt"))
        gens = {"G_AB": G_AB.state_dict(), "G_BA": G_BA.state_dict(), "args": vars(args), "epoch": epoch}
        if epoch % args.save_every == 0 or epoch == total_epochs:
            torch.save(gens, os.path.join(args.out_dir, "checkpoints", f"generators_epoch{epoch:03d}.pt"))
    fcsv.close()

    gens = {"G_AB": G_AB.state_dict(), "G_BA": G_BA.state_dict(), "args": vars(args), "epoch": total_epochs}
    torch.save(gens, os.path.join(args.out_dir, "checkpoints", "generators_final.pt"))
    peak_gb = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else float("nan")
    summary = {"parameter_counts": pc, "total_steps": step, "epochs": total_epochs,
               "training_time_min": train_time / 60,
               "train_images_per_sec": step * args.batch_size / max(train_time, 1e-9),
               "peak_gpu_memory_gb_this_session": peak_gb, "nonfinite_steps": nonfinite_total,
               "device": device_name()}
    with open(os.path.join(args.out_dir, "outputs", "train_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    plot_training(step_csv, os.path.join(args.out_dir, "outputs", "loss_curves.png"))
    log.info("summary: " + json.dumps(summary))


if __name__ == "__main__":
    main()
