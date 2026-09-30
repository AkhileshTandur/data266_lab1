#!/usr/bin/env python
"""
DATA 266 Lab 1, Task 3: translate both directions with YOUR trained generators and compute every
required metric. Pretrained Inception / LPIPS networks are used only to MEASURE images, never to make them.

    python src/evaluate_local.py --ckpt checkpoints/generators_final.pt --data_dir ../data \
        --out_dir . --real_stats ../data/real_stats.npz

Outputs (member folder):
    outputs/pred_B2A/     photo -> Monet, 256x256 RGB JPG (the Kaggle direction)
    outputs/pred_A2B/     Monet -> photo
    outputs/audit/        30 blinded input|output pairs + rating template for the human audit
    full_metrics_report.csv, submission.csv, outputs/loss_curves.png

After two raters fill in the audit template:
    python src/evaluate_local.py --mode audit_kappa --out_dir . \
        --rater1 outputs/audit/ratings_rater1.csv --rater2 outputs/audit/ratings_rater2.csv
"""
import argparse
import csv
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cyclegan import build_generator, device_name, list_images, n_params, plot_training, read_step_log  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="metrics", choices=["metrics", "audit_kappa"])
    p.add_argument("--ckpt", default="checkpoints/generators_final.pt")
    p.add_argument("--data_dir", default="../data")
    p.add_argument("--out_dir", default=".")
    p.add_argument("--real_stats", default=None, help="real_stats.npz from the Kaggle dataset")
    p.add_argument("--run_name", default="cyclegan")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_images", type=int, default=0, help="limit inputs per direction (smoke test)")
    p.add_argument("--kid_subsets", type=int, default=100)
    p.add_argument("--kid_subset_size", type=int, default=1000)
    p.add_argument("--k_nn", type=int, default=3)
    p.add_argument("--audit_n", type=int, default=30)
    p.add_argument("--audit_seed", type=int, default=266, help="keep identical across the team")
    p.add_argument("--seed", type=int, default=266)
    p.add_argument("--rater1", default=None)
    p.add_argument("--rater2", default=None)
    p.add_argument("--extractor", default="auto", choices=["auto", "pytorch_fid", "torchvision"])
    return p.parse_args()


# ----------------------------------------------------------------------------
# images and features
# ----------------------------------------------------------------------------
class Folder(Dataset):
    def __init__(self, files, norm):
        import torchvision.transforms as T
        self.files = files
        tf = [T.Resize((256, 256), interpolation=T.InterpolationMode.BICUBIC), T.ToTensor()]
        if norm:
            tf.append(T.Normalize([0.5] * 3, [0.5] * 3))
        self.tf = T.Compose(tf)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        return self.tf(Image.open(self.files[i]).convert("RGB")), i


class InceptionFeatures:
    """kind = 'pytorch_fid' (standard FID weights) or 'torchvision' (ImageNet Inception-v3)."""

    def __init__(self, device, kind="pytorch_fid"):
        self.device = device
        if kind == "pytorch_fid":
            from pytorch_fid.inception import InceptionV3
            self.model = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(device).eval()
        else:
            from torchvision.models import Inception_V3_Weights, inception_v3
            m = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True)
            m.fc = nn.Identity()
            self.model = m.to(device).eval()
            self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
            self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        self.kind = kind

    @torch.no_grad()
    def __call__(self, x01):
        x01 = x01.to(self.device)
        if self.kind == "pytorch_fid":
            return self.model(x01)[0].flatten(1).double().cpu().numpy()
        x = F.interpolate(x01, size=(299, 299), mode="bilinear", align_corners=False)
        return self.model((x - self.mean) / self.std).double().cpu().numpy()


def pick_extractor(device, monet_files, npz, choice):
    """Use the extractor whose features reproduce real_stats.npz for the real Monet images (FID close to 0)."""
    kinds = ["pytorch_fid", "torchvision"] if choice == "auto" else [choice]
    best, checks = None, {}
    for k in kinds:
        try:
            fx = InceptionFeatures(device, k)
        except Exception as e:
            print(f"extractor {k} unavailable: {e}")
            continue
        if npz is None or "mu_real" not in npz:
            return fx, checks
        f = dir_features(monet_files, fx)
        checks[k] = fid_from_stats(f.mean(0), np.cov(f, rowvar=False),
                                   npz["mu_real"].astype(np.float64), npz["sigma_real"].astype(np.float64))
        print(f"extractor check: FID(real Monet with {k}, real_stats.npz) = {checks[k]:.3f}  (should be ~0)")
        if best is None or checks[k] < checks[best[0]]:
            best = (k, fx, f)
    if best is None:
        raise SystemExit("no Inception extractor could be loaded")
    if checks[best[0]] > 5:
        print("WARNING: no extractor reproduces real_stats.npz exactly; local FID/MiFID will differ from Kaggle's. "
              "Use the course's official evaluation script for submission.csv if you have it.")
    best[1].cached_monet = best[2]
    return best[1], checks


def dir_features(files, fx, bs=64):
    dl = DataLoader(Folder(files, norm=False), batch_size=bs, num_workers=4)
    return np.concatenate([fx(x) for x, _ in dl])


# ----------------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------------
def fid_from_stats(mu1, s1, mu2, s2):
    from scipy import linalg
    diff = mu1 - mu2
    covmean = linalg.sqrtm(s1.dot(s2))
    if not np.isfinite(covmean).all():
        off = np.eye(s1.shape[0]) * 1e-6
        covmean = linalg.sqrtm((s1 + off).dot(s2 + off))
    covmean = covmean.real
    return float(diff.dot(diff) + np.trace(s1) + np.trace(s2) - 2 * np.trace(covmean))


def fid(f1, f2):
    return fid_from_stats(f1.mean(0), np.cov(f1, rowvar=False), f2.mean(0), np.cov(f2, rowvar=False))


def kid(f1, f2, n_subsets, subset_size, seed):
    rng = np.random.default_rng(seed)
    d = f1.shape[1]
    m = min(len(f1), len(f2), subset_size)
    vals = []
    for _ in range(n_subsets):
        x = f1[rng.choice(len(f1), m, replace=False)]
        y = f2[rng.choice(len(f2), m, replace=False)]
        a = (x @ x.T / d + 1) ** 3 + (y @ y.T / d + 1) ** 3
        b = (x @ y.T / d + 1) ** 3
        vals.append(((a.sum() - np.diag(a).sum()) / (m - 1) - b.sum() * 2 / m) / m)
    return float(np.mean(vals)), float(np.std(vals))


def prdc(real, fake, k, device):
    """Improved precision/recall (Kynkaanniemi 2019) and density/coverage (Naeem 2020)."""
    r = torch.from_numpy(real).float().to(device)
    f = torch.from_numpy(fake).float().to(device)

    def kth(x):
        out = []
        for i in range(0, len(x), 2048):
            out.append(torch.cdist(x[i:i + 2048], x).kthvalue(k + 1, dim=1).values)
        return torch.cat(out)
    rr, rf = kth(r), kth(f)
    prec, dens, cov_min = [], 0.0, torch.full((len(r),), float("inf"), device=device)
    for i in range(0, len(f), 2048):
        d = torch.cdist(f[i:i + 2048], r)
        inside = d <= rr[None, :]
        prec.append(inside.any(1).float())
        dens += inside.float().sum().item()
        cov_min = torch.minimum(cov_min, d.min(0).values)
    precision = torch.cat(prec).mean().item()
    rec = []
    for i in range(0, len(r), 2048):
        rec.append((torch.cdist(r[i:i + 2048], f) <= rf[None, :]).any(1).float())
    recall = torch.cat(rec).mean().item()
    density = dens / (k * len(f))
    coverage = (cov_min <= rr).float().mean().item()
    return precision, recall, density, coverage


def cos_norm(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True).clip(1e-12)


def mifid_paired(gen, real, seed):
    """As described on the competition page: mean cosine distance after subsampling to equal size."""
    rng = np.random.default_rng(seed)
    n = min(len(gen), len(real))
    g = cos_norm(gen[rng.choice(len(gen), n, replace=False)])
    r = cos_norm(real[rng.choice(len(real), n, replace=False)])
    return float(np.mean(1 - (g * r).sum(1)))


def nn_cosine_distance(gen, real):
    """Mean distance from each generated image to its closest real image (memorization check)."""
    g, r = cos_norm(gen), cos_norm(real)
    best = np.full(len(g), np.inf)
    for i in range(0, len(r), 2048):
        best = np.minimum(best, (1 - g @ r[i:i + 2048].T).min(1))
    return float(best.mean())


def find_stats(npz, domain):
    keys = list(npz.keys())
    mus = [k for k in keys if "mu" in k.lower()]
    sig = [k for k in keys if "sig" in k.lower() or "cov" in k.lower()]
    # an unlabelled single mu/sigma pair is assumed to be the Monet (Kaggle target) statistics
    pick = lambda ks: ([k for k in ks if domain in k.lower()] or (ks if len(ks) == 1 and domain == "monet" else []))
    m, s = pick(mus), pick(sig)
    if m and s:
        return npz[m[0]].astype(np.float64), npz[s[0]].astype(np.float64), f"{m[0]}/{s[0]}"
    return None


# ----------------------------------------------------------------------------
# translation
# ----------------------------------------------------------------------------
@torch.no_grad()
def translate(G, G_back, files, out_dir, bs, device, lp):
    os.makedirs(out_dir, exist_ok=True)
    dl = DataLoader(Folder(files, norm=True), batch_size=bs, num_workers=4)
    cyc, lpips_vals, n, gen_time = [], [], 0, 0.0
    for x, idx in dl:
        x = x.to(device)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        y = G(x)
        if device == "cuda":
            torch.cuda.synchronize()
        gen_time += time.time() - t0
        rec = G_back(y)
        cyc += (rec - x).abs().mean(dim=(1, 2, 3)).cpu().tolist()
        if lp is not None:
            lpips_vals += lp(x, y).flatten().cpu().tolist()
        imgs = ((y.clamp(-1, 1) * 0.5 + 0.5) * 255).round().byte().permute(0, 2, 3, 1).cpu().numpy()
        for im, i in zip(imgs, idx.tolist()):
            stem = os.path.splitext(os.path.basename(files[i]))[0]
            Image.fromarray(im, "RGB").save(os.path.join(out_dir, f"{stem}.jpg"), quality=95)
        n += len(x)
    return {"cycle_l1_mean": float(np.mean(cyc)), "cycle_l1_std": float(np.std(cyc)),
            "lpips_input_vs_output_mean": float(np.mean(lpips_vals)) if lpips_vals else float("nan"),
            "inference_images_per_sec": n / max(gen_time, 1e-9)}


def export_audit(photo_files, pred_dir, out_dir, n, seed):
    rng = np.random.default_rng(seed)
    pick = sorted(rng.choice(len(photo_files), min(n, len(photo_files)), replace=False).tolist())
    os.makedirs(out_dir, exist_ok=True)
    key_rows = []
    for i in pick:
        src = photo_files[i]
        stem = os.path.splitext(os.path.basename(src))[0]
        sid = hashlib.sha1(f"{seed}-{stem}".encode()).hexdigest()[:8]
        a = Image.open(src).convert("RGB").resize((256, 256))
        b = Image.open(os.path.join(pred_dir, f"{stem}.jpg")).convert("RGB")
        canvas = Image.new("RGB", (512, 256)); canvas.paste(a, (0, 0)); canvas.paste(b, (256, 0))
        canvas.save(os.path.join(out_dir, f"{sid}.jpg"), quality=95)
        key_rows.append((sid, os.path.basename(src)))
    rng.shuffle(key_rows)
    with open(os.path.join(out_dir, "audit_key_DO_NOT_SHOW_RATERS.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["sample_id", "source_photo"]); w.writerows(key_rows)
    with open(os.path.join(out_dir, "ratings_template.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "style_1to5", "content_1to5", "artifacts_1to5"])
        for sid, _ in key_rows:
            w.writerow([sid, "", "", ""])


def audit_kappa(args):
    from sklearn.metrics import cohen_kappa_score
    read = lambda p: {r["sample_id"]: r for r in csv.DictReader(open(p))}
    r1, r2 = read(args.rater1), read(args.rater2)
    ids = sorted(set(r1) & set(r2))
    rows = []
    for c in ("style_1to5", "content_1to5", "artifacts_1to5"):
        a = [int(r1[i][c]) for i in ids]; b = [int(r2[i][c]) for i in ids]
        rows += [(f"human_audit_{c}_mean_rater1", float(np.mean(a))),
                 (f"human_audit_{c}_mean_rater2", float(np.mean(b))),
                 (f"human_audit_{c}_pct_exact_agreement", float(np.mean(np.array(a) == np.array(b)))),
                 (f"human_audit_{c}_pct_within_1", float(np.mean(np.abs(np.array(a) - np.array(b)) <= 1))),
                 (f"human_audit_{c}_cohen_kappa", float(cohen_kappa_score(a, b))),
                 (f"human_audit_{c}_cohen_kappa_quadratic", float(cohen_kappa_score(a, b, weights="quadratic")))]
    rows.append(("human_audit_n_samples", len(ids)))
    path = os.path.join(args.out_dir, "full_metrics_report.csv")
    old = []
    if os.path.exists(path):
        old = [r for r in csv.DictReader(open(path)) if not r["metric"].startswith("human_audit")]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["metric", "direction", "value"]); w.writeheader()
        w.writerows(old)
        for k, v in rows:
            w.writerow({"metric": k, "direction": "B2A (photo->monet)", "value": f"{v:.4g}"})
    for k, v in rows:
        print(f"{k}: {v:.4g}")


# ----------------------------------------------------------------------------
def main():
    args = parse_args()
    if args.mode == "audit_kappa":
        audit_kappa(args); return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location=device)
    G_AB, G_BA = build_generator(ck["args"]).to(device).eval(), build_generator(ck["args"]).to(device).eval()
    G_AB.load_state_dict(ck["G_AB"]); G_BA.load_state_dict(ck["G_BA"])
    print(f"loaded {args.ckpt} (epoch {ck.get('epoch')}) on {device_name()}")
    try:
        import lpips
        lp = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    except ImportError:
        lp = None
        print("lpips not installed: pip install lpips  (LPIPS will be NaN)")
    monet = list_images(os.path.join(args.data_dir, "monet_jpg"))
    photo = list_images(os.path.join(args.data_dir, "photo_jpg"))
    real_npz = np.load(args.real_stats) if args.real_stats and os.path.exists(args.real_stats) else None
    if real_npz is not None:
        print("real_stats.npz keys:", {k: real_npz[k].shape for k in real_npz.keys()})
    fx, ext_checks = pick_extractor(device, monet, real_npz, args.extractor)
    print("feature extractor:", fx.kind)
    if args.max_images:
        monet, photo = monet[:args.max_images], photo[:args.max_images]
    out = os.path.join(args.out_dir, "outputs")
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    dirs = {"B2A": ("photo->monet", photo, monet, G_BA, G_AB, os.path.join(out, "pred_B2A"), "monet"),
            "A2B": ("monet->photo", monet, photo, G_AB, G_BA, os.path.join(out, "pred_A2B"), "photo")}
    feats = {}
    if getattr(fx, "cached_monet", None) is not None and not args.max_images:
        feats["monet"] = fx.cached_monet
    rows, sub = [], {}
    for key, (label, inp, target, G, Gb, pred_dir, dom) in dirs.items():
        print(f"\n== {label}: translating {len(inp)} images ==")
        t = translate(G, Gb, inp, pred_dir, args.batch_size, device, lp)
        gen_files = [os.path.join(pred_dir, os.path.splitext(os.path.basename(f))[0] + ".jpg") for f in inp]
        for name, files in (("monet", monet), ("photo", photo)):
            if name not in feats:
                feats[name] = dir_features(files, fx)
        f_in = feats["photo"] if dom == "monet" else feats["monet"]
        f_real = feats[dom]
        f_gen = dir_features(gen_files, fx)
        m = {"FID": fid(f_gen, f_real)}
        if real_npz is not None:
            st = find_stats(real_npz, dom)
            if st:
                m[f"FID_vs_real_stats_npz[{st[2]}]"] = fid_from_stats(f_gen.mean(0), np.cov(f_gen, rowvar=False), st[0], st[1])
            if dom == "monet" and "feats_real" in real_npz:
                m["MiFID_cosine_paired_vs_npz_feats"] = mifid_paired(f_gen, real_npz["feats_real"].astype(np.float64), args.seed)
        m["KID_mean"], m["KID_std"] = kid(f_gen, f_real, args.kid_subsets, args.kid_subset_size, args.seed)
        p, r, d, c = prdc(f_real, f_gen, args.k_nn, device)
        m.update({"precision": p, "recall": r, "density": d, "coverage": c})
        m["MiFID_cosine_paired"] = mifid_paired(f_gen, f_real, args.seed)
        m["nearest_real_cosine_distance_mean"] = nn_cosine_distance(f_gen, f_real)
        m["content_cosine_similarity_input_vs_output"] = float(np.mean((cos_norm(f_in[:len(f_gen)]) * cos_norm(f_gen)).sum(1)))
        m.update(t)
        m["n_generated"] = len(f_gen)
        for k, v in m.items():
            rows.append((k, f"{key} ({label})", v))
            print(f"  {k}: {v:.5g}")
        if key == "B2A":
            npz_fid = [v for k, v in m.items() if k.startswith("FID_vs_real_stats_npz")]
            sub = {"FID": npz_fid[0] if npz_fid else m["FID"],
                   "MiFID": m.get("MiFID_cosine_paired_vs_npz_feats", m["MiFID_cosine_paired"])}

    # training stability + cost from the training log
    step_csv = os.path.join(args.out_dir, "logs", f"train_steps_{args.run_name}.csv")
    if os.path.exists(step_csv):
        d = read_step_log(step_csv)
        last = d["epoch"] == d["epoch"].max()
        fin = lambda v: v[np.isfinite(v)]
        for k, v in (("cycle_loss_A_last_epoch_mean", d["cyc_A"][last].mean()),
                     ("cycle_loss_B_last_epoch_mean", d["cyc_B"][last].mean()),
                     ("identity_loss_A_last_epoch_mean", d["idt_A"][last].mean()),
                     ("identity_loss_B_last_epoch_mean", d["idt_B"][last].mean()),
                     ("G_loss_last_epoch_mean", d["loss_G"][last].mean()),
                     ("D_A_loss_last_epoch_mean", d["D_A"][last].mean()),
                     ("D_B_loss_last_epoch_mean", d["D_B"][last].mean()),
                     ("grad_norm_G_mean", fin(d["gnorm_G"]).mean()), ("grad_norm_G_max", fin(d["gnorm_G"]).max()),
                     ("grad_norm_D_mean", fin(d["gnorm_D"]).mean()), ("grad_norm_D_max", fin(d["gnorm_D"]).max()),
                     ("nan_or_nonfinite_steps", d["nonfinite"].sum()), ("total_steps", len(d["step"]))):
            rows.append((k, "training", float(v)))
        plot_training(step_csv, os.path.join(out, "loss_curves.png"))
    summ = os.path.join(out, "train_summary.json")
    if os.path.exists(summ):
        s = json.load(open(summ))
        for k, v in s["parameter_counts"].items():
            rows.append((f"parameter_count_{k}", "model", v))
        for k in ("training_time_min", "train_images_per_sec", "peak_gpu_memory_gb_this_session", "device"):
            rows.append((k, "training", s[k]))
    else:
        rows.append(("parameter_count_G_AB", "model", n_params(G_AB)))
    if device == "cuda":
        rows.append(("peak_gpu_memory_gb_eval", "evaluation", torch.cuda.max_memory_allocated() / 1e9))
    rows.append(("feature_extractor", "evaluation", fx.kind))
    for k, v in ext_checks.items():
        rows.append((f"extractor_check_FID_real_monet_vs_npz[{k}]", "evaluation", v))
    rows.append(("checkpoint", "evaluation", os.path.basename(args.ckpt)))

    with open(os.path.join(args.out_dir, "full_metrics_report.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["metric", "direction", "value"])
        for k, dname, v in rows:
            w.writerow([k, dname, f"{v:.6g}" if isinstance(v, (float, np.floating)) else v])
    with open(os.path.join(args.out_dir, "submission.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["FID", "MiFID"]); w.writerow([f"{sub['FID']:.6f}", f"{sub['MiFID']:.6f}"])
    export_audit(photo, dirs["B2A"][5], os.path.join(out, "audit"), args.audit_n, args.audit_seed)
    print("\nwrote full_metrics_report.csv, submission.csv, outputs/pred_*, outputs/audit/")
    print("CHECK submission.csv columns against the course's sample submission / evaluation script before uploading.")


if __name__ == "__main__":
    main()
