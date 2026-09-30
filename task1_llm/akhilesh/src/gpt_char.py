#!/usr/bin/env python
"""
DATA 266 Lab 1, Task 1: character-level GPT built from scratch on TinyStories.

Attention, causal masking, LayerNorm, blocks and the LM head are written by hand.
Nothing from nn.Transformer*, nn.MultiheadAttention, nn.LayerNorm or
F.scaled_dot_product_attention is used.

Run from your member folder (task1_llm/<your_name>/):
    python src/gpt_char.py --config src/config.yaml --out_dir .
Resume after a lab session ends:
    python src/gpt_char.py --config src/config.yaml --out_dir . --resume checkpoints/last.pt
Generate only, from a saved checkpoint:
    python src/gpt_char.py --out_dir . --generate_only checkpoints/best.pt
"""
import argparse
import csv
import json
import logging
import math
import os
import platform
import random
import re
import subprocess
import sys
import time
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

EOS, UNK = "\x03", "\x01"
DEFAULT_PROMPTS = [
    "Once upon a time",
    "One day, a little girl named Lily",
    "The dog was very sad because",
    "Tom and his mom went to the park.",
    "There was a big red ball",
]


# ----------------------------------------------------------------------------
# config, logging, manifest
# ----------------------------------------------------------------------------
def load_cfg(path):
    with open(path) as f:
        if path.endswith((".yaml", ".yml")):
            import yaml
            return yaml.safe_load(f) or {}
        return json.load(f)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    p.add_argument("--out_dir", default=".")
    p.add_argument("--run_name", default="gpt_char")
    p.add_argument("--seed", type=int, default=266)
    p.add_argument("--cache_dir", default=None, help="HF datasets cache (optional)")
    # data
    p.add_argument("--dataset", default="roneneldan/TinyStories")
    p.add_argument("--n_train", type=int, default=100_000)
    p.add_argument("--n_val", type=int, default=10_000)
    p.add_argument("--min_char_freq", type=int, default=50)
    p.add_argument("--block_size", type=int, default=256)
    # model
    p.add_argument("--n_layer", type=int, default=6)
    p.add_argument("--n_head", type=int, default=6)
    p.add_argument("--n_embd", type=int, default=384)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--tie_weights", type=int, default=1)
    # optimisation
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--min_lr", type=float, default=6e-5)
    p.add_argument("--warmup_steps", type=int, default=1000)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--amp", type=int, default=1)
    p.add_argument("--eval_batches", type=int, default=0, help="0 = full validation set")
    p.add_argument("--log_every", type=int, default=200)
    # generation
    p.add_argument("--gen_len", type=int, default=600)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=0)
    p.add_argument("--prompts", nargs="+", default=DEFAULT_PROMPTS)
    # control
    p.add_argument("--resume", default=None)
    p.add_argument("--generate_only", default=None)
    pre, _ = p.parse_known_args()
    if pre.config:
        p.set_defaults(**load_cfg(pre.config))
    return p.parse_args()


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def device_name():
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return platform.processor() or "CPU"


def setup_run(args):
    for d in ("checkpoints", "outputs", "logs", "data_processed"):
        os.makedirs(os.path.join(args.out_dir, d), exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(args.out_dir, "logs", f"{args.run_name}_{stamp}.log")
    logger = logging.getLogger(args.run_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(message)s")
    for h in (logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)):
        h.setFormatter(fmt)
        logger.addHandler(h)
    manifest = {
        "run_name": args.run_name, "timestamp": stamp, "args": vars(args),
        "python": sys.version, "platform": platform.platform(),
        "torch": torch.__version__, "cuda": torch.version.cuda, "device": device_name(),
    }
    try:
        manifest["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        manifest["git_commit"] = None
    with open(os.path.join(args.out_dir, "logs", f"manifest_{args.run_name}_{stamp}.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    try:
        freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"]).decode()
        with open(os.path.join(args.out_dir, "logs", f"pip_freeze_{stamp}.txt"), "w") as f:
            f.write(freeze)
    except Exception:
        pass
    logger.info(f"log file: {log_path}")
    logger.info(f"device: {device_name()}")
    logger.info("args: " + json.dumps(vars(args)))
    return logger, stamp


# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------
class CharTokenizer:
    def __init__(self, itos):
        self.itos = list(itos)
        self.stoi = {c: i for i, c in enumerate(self.itos)}
        self._lut = None

    @classmethod
    def build(cls, text, min_freq):
        cps = np.frombuffer(text.encode("utf-32-le"), dtype=np.uint32)
        vals, counts = np.unique(cps, return_counts=True)
        chars = sorted(chr(v) for v, n in zip(vals, counts)
                       if n >= min_freq and chr(v) not in (EOS, UNK))
        return cls([EOS, UNK] + chars)

    def encode(self, text):
        if self._lut is None:
            self._lut = np.full(0x110000, self.stoi[UNK], dtype=np.int16)
            for c, i in self.stoi.items():
                self._lut[ord(c)] = i
        cps = np.frombuffer(text.encode("utf-32-le"), dtype=np.uint32)
        return self._lut[cps]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)

    @property
    def vocab_size(self):
        return len(self.itos)


def load_stories(args, log):
    from datasets import load_dataset
    ds = load_dataset(args.dataset, split="train", cache_dir=args.cache_dir)
    extra = 2000  # headroom for empty rows
    ds = ds.shuffle(seed=args.seed).select(range(args.n_train + args.n_val + extra))
    texts = [t.strip() for t in ds["text"] if isinstance(t, str) and t.strip()]
    train = texts[:args.n_train]
    val = texts[args.n_train:args.n_train + args.n_val]
    log.info(f"stories: train={len(train)} val={len(val)}")
    return train, val


class Blocks:
    """Non-overlapping fixed-length (input, target) windows over one long token stream."""

    def __init__(self, data, block):
        self.data = torch.from_numpy(data.astype(np.int16))
        self.block = block
        self.n = (len(data) - 1) // block
        self.starts = torch.arange(self.n) * block
        self.offs = torch.arange(block + 1)

    def batch(self, idx):
        chunk = self.data[self.starts[idx][:, None] + self.offs].long()
        return chunk[:, :-1], chunk[:, 1:]


# ----------------------------------------------------------------------------
# model (all written by hand)
# ----------------------------------------------------------------------------
class LayerNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.bias = nn.Parameter(torch.zeros(d))
        self.eps = eps

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(var + self.eps) * self.weight + self.bias


class CausalSelfAttention(nn.Module):
    def __init__(self, d, n_head, block, dropout):
        super().__init__()
        assert d % n_head == 0
        self.n_head = n_head
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
        mask = torch.tril(torch.ones(block, block, dtype=torch.bool))
        self.register_buffer("mask", mask.view(1, 1, block, block), persistent=False)

    def forward(self, x):
        B, T, C = x.shape
        hd = C // self.n_head
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, hd).transpose(1, 2)
        k = k.view(B, T, self.n_head, hd).transpose(1, 2)
        v = v.view(B, T, self.n_head, hd).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(hd)
        att = att.float().masked_fill(~self.mask[:, :, :T, :T], float("-inf"))
        att = F.softmax(att, dim=-1).to(v.dtype)
        att = self.attn_drop(att)
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))


class FeedForward(nn.Module):
    def __init__(self, d, dropout):
        super().__init__()
        self.fc1 = nn.Linear(d, 4 * d)
        self.fc2 = nn.Linear(4 * d, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.fc2(F.gelu(self.fc1(x))))


class Block(nn.Module):
    """Pre-LN block: x + MHSA(LN(x)), then x + FFN(LN(x))."""

    def __init__(self, d, n_head, block, dropout):
        super().__init__()
        self.ln1 = LayerNorm(d)
        self.attn = CausalSelfAttention(d, n_head, block, dropout)
        self.ln2 = LayerNorm(d)
        self.ffn = FeedForward(d, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, block_size, n_layer, n_head, n_embd, dropout, tie_weights=True):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)])
        self.ln_f = LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        if tie_weights:
            self.lm_head.weight = self.tok_emb.weight
        self.apply(self._init)
        for name, p in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("fc2.weight"):
                nn.init.normal_(p, 0.0, 0.02 / math.sqrt(2 * n_layer))

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def forward(self, idx):
        B, T = idx.shape
        assert T <= self.block_size
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for blk in self.blocks:
            x = blk(x)
        return self.lm_head(self.ln_f(x))


@torch.no_grad()
def generate(model, idx, n_new, temperature=1.0, top_k=0, greedy=False):
    model.eval()
    for _ in range(n_new):
        logits = model(idx[:, -model.block_size:])[:, -1, :].float()
        if greedy:
            nxt = logits.argmax(-1, keepdim=True)
        else:
            logits = logits / max(temperature, 1e-6)
            if top_k and top_k > 0:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float("-inf")
            nxt = torch.multinomial(F.softmax(logits, dim=-1), 1)
        idx = torch.cat([idx, nxt], dim=1)
    return idx


# ----------------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------------
def words(text):
    return re.findall(r"[a-z']+", text.lower())


def distinct_n(texts, n):
    grams, total = set(), 0
    for t in texts:
        w = words(t)
        g = [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]
        grams.update(g)
        total += len(g)
    return len(grams) / max(total, 1)


def repeated_4gram_rate(texts):
    rates = []
    for t in texts:
        w = words(t)
        g = [tuple(w[i:i + 4]) for i in range(len(w) - 3)]
        if not g:
            continue
        seen, rep = set(), 0
        for x in g:
            rep += x in seen
            seen.add(x)
        rates.append(rep / len(g))
    return float(np.mean(rates)) if rates else 0.0


@torch.no_grad()
def evaluate(model, blocks, idxs, bs, device, amp_ctx):
    model.eval()
    loss_sum, correct, n = 0.0, 0, 0
    for b in range(0, len(idxs), bs):
        x, y = blocks.batch(idxs[b:b + bs])
        x, y = x.to(device), y.to(device)
        with amp_ctx():
            logits = model(x)
        logits = logits.float()
        loss_sum += F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum").item()
        correct += (logits.argmax(-1) == y).sum().item()
        n += y.numel()
    model.train()
    return loss_sum / n, correct / n


def plot_curves(history, step_log, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ep = [h["epoch"] for h in history]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    ax[0].plot(ep, [h["train_loss_eval"] for h in history], "o-", label="train (eval mode)")
    ax[0].plot(ep, [h["train_loss_running"] for h in history], "s--", label="train (running, dropout on)")
    ax[0].plot(ep, [h["val_loss"] for h in history], "o-", label="validation")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("cross-entropy (nats/char)"); ax[0].legend(); ax[0].set_title("Loss per epoch")
    steps = np.array([r[0] for r in step_log]); losses = np.array([r[3] for r in step_log])
    gns = np.array([r[4] for r in step_log])
    k = 100
    if len(losses) > k:
        sm = np.convolve(losses, np.ones(k) / k, mode="valid")
        ax[1].plot(steps[k - 1:], sm)
    else:
        ax[1].plot(steps, losses)
    ax[1].set_xlabel("step"); ax[1].set_ylabel("train loss (100-step avg)"); ax[1].set_title("Training loss per step")
    ax[2].plot(steps, gns, lw=0.5)
    ax[2].set_yscale("log"); ax[2].set_xlabel("step"); ax[2].set_ylabel("grad norm (pre-clip)"); ax[2].set_title("Gradient norm")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "outputs", "loss_curves.png"), dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def build_model(a, vocab_size):
    return GPT(vocab_size, a["block_size"], a["n_layer"], a["n_head"], a["n_embd"],
               a["dropout"], bool(a["tie_weights"]))


def run_generation(model, tok, args, device, log):
    samples, greedy_samples = [], []
    n_tok, t0 = 0, time.time()
    if device == "cuda":
        torch.cuda.synchronize()
    for p in args.prompts:
        idx = torch.from_numpy(tok.encode(p).astype(np.int64))[None].to(device)
        out = generate(model, idx, args.gen_len, args.temperature, args.top_k)
        samples.append({"prompt": p, "mode": f"temp={args.temperature},top_k={args.top_k}",
                        "text": tok.decode(out[0].tolist()).replace(EOS, "\n<EOS>\n").replace(UNK, "<unk>")})
        n_tok += args.gen_len
    if device == "cuda":
        torch.cuda.synchronize()
    gen_tps = n_tok / (time.time() - t0)
    for p in args.prompts[:2]:
        idx = torch.from_numpy(tok.encode(p).astype(np.int64))[None].to(device)
        out = generate(model, idx, args.gen_len, greedy=True)
        greedy_samples.append({"prompt": p, "mode": "greedy",
                               "text": tok.decode(out[0].tolist()).replace(EOS, "\n<EOS>\n").replace(UNK, "<unk>")})
    with open(os.path.join(args.out_dir, "outputs", "samples.json"), "w") as f:
        json.dump(samples + greedy_samples, f, indent=2)
    with open(os.path.join(args.out_dir, "outputs", "samples.txt"), "w") as f:
        for s in samples + greedy_samples:
            f.write(f"=== prompt: {s['prompt']!r} | {s['mode']} ===\n{s['text']}\n\n")
    texts = [s["text"] for s in samples]
    gen_metrics = {
        "distinct_1": distinct_n(texts, 1), "distinct_2": distinct_n(texts, 2),
        "distinct_3": distinct_n(texts, 3),
        "repeated_4gram_rate": repeated_4gram_rate(texts),
        "repeated_4gram_rate_greedy": repeated_4gram_rate([s["text"] for s in greedy_samples]),
        "gen_tokens_per_sec": gen_tps,
    }
    log.info("generation metrics: " + json.dumps(gen_metrics))
    return gen_metrics


def main():
    args = parse_args()
    set_seed(args.seed)
    log, stamp = setup_run(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = bool(args.amp) and device == "cuda"
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    amp_ctx = (lambda: torch.autocast("cuda", dtype=amp_dtype)) if use_amp else nullcontext

    # ---- generate-only mode ----
    if args.generate_only:
        ck = torch.load(args.generate_only, map_location=device)
        tok = CharTokenizer(ck["itos"])
        model = build_model(ck["args"], tok.vocab_size).to(device)
        model.load_state_dict(ck["model"])
        run_generation(model, tok, args, device, log)
        return

    # ---- data ----
    t_data = time.time()
    train_s, val_s = load_stories(args, log)
    train_text = EOS.join(train_s) + EOS
    val_text = EOS.join(val_s) + EOS
    tok = CharTokenizer.build(train_text, args.min_char_freq)
    with open(os.path.join(args.out_dir, "data_processed", "vocab.json"), "w") as f:
        json.dump({"itos": tok.itos, "char_to_idx": tok.stoi}, f, ensure_ascii=False, indent=1)
    train_ids, val_ids = tok.encode(train_text), tok.encode(val_text)
    unk_rate = float((val_ids == tok.stoi[UNK]).mean())
    log.info(f"vocab={tok.vocab_size} train_chars={len(train_ids):,} val_chars={len(val_ids):,} "
             f"val_unk_rate={unk_rate:.5f} (data prep {time.time() - t_data:.0f}s)")
    train_b, val_b = Blocks(train_ids, args.block_size), Blocks(val_ids, args.block_size)
    log.info(f"sequences: train={train_b.n:,} val={val_b.n:,} of length {args.block_size}")

    g = torch.Generator().manual_seed(args.seed)
    val_idx = torch.arange(val_b.n)
    train_eval_idx = torch.randperm(train_b.n, generator=g)[:val_b.n]
    if args.eval_batches > 0:
        val_idx = val_idx[:args.eval_batches * args.batch_size]
        train_eval_idx = train_eval_idx[:args.eval_batches * args.batch_size]

    # ---- model / optimiser ----
    model = build_model(vars(args), tok.vocab_size).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"parameters: {n_params:,}")
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt_kw = dict(lr=args.lr, betas=(0.9, args.beta2))
    if device == "cuda":
        opt_kw["fused"] = True
    try:
        opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.weight_decay},
                                 {"params": no_decay, "weight_decay": 0.0}], **opt_kw)
    except TypeError:
        opt_kw.pop("fused", None)
        opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.weight_decay},
                                 {"params": no_decay, "weight_decay": 0.0}], **opt_kw)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)

    steps_per_epoch = math.ceil(train_b.n / args.batch_size)
    total_steps = steps_per_epoch * args.epochs

    def lr_at(step):
        if step < args.warmup_steps:
            return args.lr * (step + 1) / args.warmup_steps
        prog = min(1.0, (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps))
        return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * prog))

    step, start_epoch, best_val, best_epoch = 0, 1, float("inf"), 0
    history, counters = [], {"nan_or_inf_loss": 0, "nonfinite_grad": 0, "loss_spikes": 0}
    train_time = 0.0
    step_csv = os.path.join(args.out_dir, "logs", f"train_steps_{args.run_name}.csv")
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        if ck.get("scaler"):
            scaler.load_state_dict(ck["scaler"])
        step, start_epoch = ck["step"], ck["epoch"] + 1
        best_val, best_epoch = ck["best_val"], ck["best_epoch"]
        history, counters, train_time = ck["history"], ck["counters"], ck["train_time"]
        log.info(f"resumed from {args.resume} at epoch {ck['epoch']} step {step}")
    new_csv = not os.path.exists(step_csv)
    fcsv = open(step_csv, "a", newline="")
    wcsv = csv.writer(fcsv)
    if new_csv:
        wcsv.writerow(["step", "epoch", "lr", "loss", "grad_norm"])
    log.info(f"steps/epoch={steps_per_epoch} total_steps={total_steps} amp={use_amp}({amp_dtype})")

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    ema = None
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        perm = torch.randperm(train_b.n)
        loss_sum, tok_count = 0.0, 0
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for bi in range(0, train_b.n, args.batch_size):
            x, y = train_b.batch(perm[bi:bi + args.batch_size])
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            lr = lr_at(step)
            for gp in opt.param_groups:
                gp["lr"] = lr
            with amp_ctx():
                logits = model(x)
            loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), y.reshape(-1))
            lv = loss.item()
            if not math.isfinite(lv):
                counters["nan_or_inf_loss"] += 1
                opt.zero_grad(set_to_none=True)
                log.info(f"non-finite loss at step {step}, skipping")
                step += 1
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip).item()
            if not math.isfinite(gn):
                counters["nonfinite_grad"] += 1
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            if ema is not None and step > args.warmup_steps and lv > 1.5 * ema:
                counters["loss_spikes"] += 1
            ema = lv if ema is None else 0.99 * ema + 0.01 * lv
            loss_sum += lv * y.numel()
            tok_count += y.numel()
            wcsv.writerow([step, epoch, f"{lr:.3e}", f"{lv:.5f}", f"{gn:.4f}"])
            if step % args.log_every == 0:
                el = time.time() - t0
                log.info(f"ep {epoch} step {step}/{total_steps} lr {lr:.2e} loss {lv:.4f} "
                         f"gnorm {gn:.3f} | {tok_count / max(el, 1e-9):,.0f} tok/s")
                fcsv.flush()
            step += 1
        if device == "cuda":
            torch.cuda.synchronize()
        ep_time = time.time() - t0
        train_time += ep_time
        val_loss, val_acc = evaluate(model, val_b, val_idx, args.batch_size, device, amp_ctx)
        tr_loss, tr_acc = evaluate(model, train_b, train_eval_idx, args.batch_size, device, amp_ctx)
        h = {"epoch": epoch, "train_loss_running": loss_sum / max(tok_count, 1), "train_loss_eval": tr_loss,
             "train_acc": tr_acc, "val_loss": val_loss, "val_acc": val_acc,
             "val_ppl": math.exp(val_loss), "val_bpc": val_loss / math.log(2),
             "epoch_time_s": ep_time, "train_tok_per_s": tok_count / ep_time}
        history.append(h)
        log.info("EPOCH " + json.dumps({k: round(v, 5) if isinstance(v, float) else v for k, v in h.items()}))
        ck = {"model": model.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(),
              "itos": tok.itos, "args": vars(args), "epoch": epoch, "step": step,
              "best_val": best_val, "best_epoch": best_epoch, "history": history,
              "counters": counters, "train_time": train_time}
        if val_loss < best_val:
            best_val, best_epoch = val_loss, epoch
            ck["best_val"], ck["best_epoch"] = best_val, best_epoch
            torch.save({"model": model.state_dict(), "itos": tok.itos, "args": vars(args),
                        "epoch": epoch, "val_loss": val_loss}, os.path.join(args.out_dir, "checkpoints", "best.pt"))
            log.info(f"new best val {val_loss:.4f} -> checkpoints/best.pt")
        torch.save(ck, os.path.join(args.out_dir, "checkpoints", "last.pt"))
    fcsv.close()

    # ---- final evaluation, generation, reports ----
    best = torch.load(os.path.join(args.out_dir, "checkpoints", "best.pt"), map_location=device)
    model.load_state_dict(best["model"])
    val_loss, val_acc = evaluate(model, val_b, val_idx, args.batch_size, device, amp_ctx)
    tr_loss, tr_acc = evaluate(model, train_b, train_eval_idx, args.batch_size, device, amp_ctx)
    peak_gb = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else float("nan")
    gen = run_generation(model, tok, args, device, log)

    step_log = []
    with open(step_csv) as f:
        r = csv.reader(f); next(r)
        for row in r:
            step_log.append((int(row[0]), int(row[1]), float(row[2]), float(row[3]), float(row[4])))
    plot_curves(history, step_log, args.out_dir)
    gns = np.array([s[4] for s in step_log if math.isfinite(s[4])])
    tok_per_epoch = train_b.n * args.block_size

    metrics = [
        ("train_cross_entropy_loss", tr_loss),
        ("train_cross_entropy_loss_running_last_epoch", history[-1]["train_loss_running"]),
        ("validation_cross_entropy_loss", val_loss),
        ("train_perplexity", math.exp(tr_loss)),
        ("validation_perplexity", math.exp(val_loss)),
        ("validation_bits_per_character", val_loss / math.log(2)),
        ("generalization_gap_val_minus_train", val_loss - tr_loss),
        ("top1_next_char_accuracy_val", val_acc),
        ("top1_next_char_accuracy_train", tr_acc),
        ("distinct_1_word", gen["distinct_1"]),
        ("distinct_2_word", gen["distinct_2"]),
        ("distinct_3_word", gen["distinct_3"]),
        ("repeated_4gram_rate_word", gen["repeated_4gram_rate"]),
        ("repeated_4gram_rate_word_greedy", gen["repeated_4gram_rate_greedy"]),
        ("grad_norm_mean", float(gns.mean())),
        ("grad_norm_p99", float(np.percentile(gns, 99))),
        ("grad_norm_max", float(gns.max())),
        ("loss_spikes_gt_1.5x_ema", counters["loss_spikes"]),
        ("nan_or_inf_loss_steps", counters["nan_or_inf_loss"]),
        ("nonfinite_grad_steps", counters["nonfinite_grad"]),
        ("parameter_count", n_params),
        ("training_tokens_per_sec", tok_per_epoch * len(history) / max(train_time, 1e-9)),
        ("generation_tokens_per_sec", gen["gen_tokens_per_sec"]),
        ("peak_gpu_memory_gb", peak_gb),
        ("total_training_time_min", train_time / 60),
        ("epochs_trained", len(history)),
        ("best_epoch", best_epoch),
        ("vocab_size", tok.vocab_size),
        ("device", device_name()),
    ]
    with open(os.path.join(args.out_dir, "metrics_report.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["metric", "value"])
        for k, v in metrics:
            w.writerow([k, f"{v:.6g}" if isinstance(v, float) else v])
    with open(os.path.join(args.out_dir, "outputs", "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    log.info("final metrics: " + json.dumps({k: v for k, v in metrics}, default=str))

    fa = os.path.join(args.out_dir, "failure_analysis.md")
    if not os.path.exists(fa):
        with open(fa, "w") as f:
            f.write("# Task 1 failure analysis\n\nSamples: `outputs/samples.txt` (checkpoint `checkpoints/best.pt`).\n\n")
            for i in range(1, 4):
                f.write(f"## Failure case {i}\n\n**Snippet:**\n\n```\n\n```\n\n**Failure type:** \n\n**Observation:** \n\n")
    rm = os.path.join(args.out_dir, "results.md")
    if not os.path.exists(rm):
        with open(rm, "w") as f:
            f.write("# Task 1 results: character-level GPT\n\n## Architecture and hyperparameters\n\n| setting | value |\n|---|---|\n")
            for k in ("n_layer", "n_head", "n_embd", "block_size", "dropout", "tie_weights", "batch_size", "lr",
                      "min_lr", "warmup_steps", "weight_decay", "epochs", "temperature", "top_k"):
                f.write(f"| {k} | {getattr(args, k)} |\n")
            f.write(f"| parameters | {n_params:,} |\n\n## Why these choices\n\n\n## Metrics\n\n| metric | value |\n|---|---|\n")
            for k, v in metrics:
                f.write(f"| {k} | {v:.4g} |\n" if isinstance(v, float) else f"| {k} | {v} |\n")
            f.write("\nLoss curves: `outputs/loss_curves.png`. Raw log: `logs/`.\n")
    log.info("done")


if __name__ == "__main__":
    main()
