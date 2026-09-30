#!/usr/bin/env python
"""
DATA 266 Lab 1, Task 2: Yelp Polarity sentiment classification, embeddings learned from scratch.

Models (pick your own 3, and make sure no teammate uses the same set + hyperparameters):
    bow     mean-pooled embeddings + MLP (fastText-style baseline)
    cnn     Kim-style 1D CNN, several kernel widths, max-over-time pooling
    bilstm  bidirectional LSTM (last-state or max pooling)
    bigru   bidirectional GRU

Run from your member folder (task2_sentiment/<your_name>/):
    python src/sentiment.py --config src/cfg_baseline.yaml --out_dir .
    python src/sentiment.py --config src/cfg_exp1.yaml --out_dir .
    python src/sentiment.py --config src/cfg_exp2.yaml --out_dir .
    python src/sentiment.py --mode compare --out_dir . \
        --baseline_preds outputs/baseline/test_preds.npz \
        --exp_preds outputs/exp1/test_preds.npz outputs/exp2/test_preds.npz
Preprocessing is cached in data_processed/ and reused by every run with the same settings.
"""
import argparse
import csv
import hashlib
import json
import logging
import math
import os
import pickle
import platform
import random
import re
import subprocess
import sys
import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


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
    p.add_argument("--mode", default="train", choices=["train", "compare"])
    p.add_argument("--out_dir", default=".")
    p.add_argument("--run_name", default="baseline")
    p.add_argument("--seed", type=int, default=266)
    p.add_argument("--cache_dir", default=None)
    # data / preprocessing
    p.add_argument("--n_train", type=int, default=120_000, help="subsample of the 560K train split")
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--remove_stopwords", type=int, default=1)
    p.add_argument("--keep_negations", type=int, default=1)
    p.add_argument("--stem", type=int, default=1)
    p.add_argument("--min_freq", type=int, default=3)
    p.add_argument("--max_vocab", type=int, default=30_000)
    p.add_argument("--max_len", type=int, default=256)
    # model
    p.add_argument("--model", default="bow", choices=["bow", "cnn", "bilstm", "bigru"])
    p.add_argument("--emb_dim", type=int, default=128)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--num_layers", type=int, default=1)
    p.add_argument("--kernels", type=int, nargs="+", default=[3, 4, 5])
    p.add_argument("--n_filters", type=int, default=100)
    p.add_argument("--pooling", default="last", choices=["last", "max"])
    p.add_argument("--dropout", type=float, default=0.3)
    # training
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--n_boot", type=int, default=1000)
    p.add_argument("--num_workers", type=int, default=2)
    # compare mode
    p.add_argument("--baseline_preds", default=None)
    p.add_argument("--exp_preds", nargs="*", default=[])
    pre, _ = p.parse_known_args()
    if pre.config:
        p.set_defaults(**load_cfg(pre.config))
    return p.parse_args()


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def device_name():
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return platform.processor() or "CPU"


def setup_run(args):
    for d in ("checkpoints", "outputs", "logs", "data_processed"):
        os.makedirs(os.path.join(args.out_dir, d), exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(args.out_dir, "logs", f"{args.run_name}_{args.mode}_{stamp}.log")
    logger = logging.getLogger(args.run_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(message)s")
    for h in (logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)):
        h.setFormatter(fmt); logger.addHandler(h)
    manifest = {"run_name": args.run_name, "mode": args.mode, "timestamp": stamp, "args": vars(args),
                "python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
                "cuda": torch.version.cuda, "device": device_name()}
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
    logger.info(f"log file: {log_path} | device: {device_name()}")
    logger.info("args: " + json.dumps(vars(args)))
    return logger


# ----------------------------------------------------------------------------
# preprocessing
# ----------------------------------------------------------------------------
NEGATIONS = {"no", "not", "nor", "never", "none", "nothing", "nobody", "neither", "nowhere", "cannot"}
NEG_RE = re.compile(r"\b(not|no|never|nothing|none|cannot)\b|n't")
NT_RE = re.compile(r"n't\b")
NONALPHA_RE = re.compile(r"[^a-z\s]")


def make_cleaner(args):
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    stop = set(ENGLISH_STOP_WORDS) if args.remove_stopwords else set()
    if args.keep_negations:
        stop -= NEGATIONS
    stem = None
    if args.stem:
        try:
            from functools import lru_cache
            from nltk.stem import PorterStemmer
            ps = PorterStemmer()
            stem = lru_cache(maxsize=500_000)(ps.stem)
        except ImportError:
            print("nltk not installed, stemming disabled (pip install nltk)")

    def clean(text):
        t = text.lower().replace("\\n", " ").replace('\\"', " ")
        t = NT_RE.sub(" not", t)
        t = NONALPHA_RE.sub(" ", t)
        toks = [w for w in t.split() if len(w) > 1 and w not in stop]
        if stem:
            toks = [stem(w) for w in toks]
        return toks
    return clean


def load_yelp(cache_dir):
    from datasets import load_dataset
    last = None
    for name in ("fancyzhx/yelp_polarity", "yelp_polarity"):
        try:
            return load_dataset(name, cache_dir=cache_dir)
        except Exception as e:
            last = e
    raise last


def encode(tok_lists, stoi, max_len):
    ids = np.zeros((len(tok_lists), max_len), dtype=np.int64)
    lens = np.zeros(len(tok_lists), dtype=np.int64)
    for i, t in enumerate(tok_lists):
        s = [stoi.get(w, 1) for w in t[:max_len]] or [1]
        ids[i, :len(s)] = s
        lens[i] = len(s)
    return ids, lens


def prepare_data(args, log):
    keys = ["n_train", "val_frac", "remove_stopwords", "keep_negations", "stem", "min_freq",
            "max_vocab", "max_len", "seed"]
    h = hashlib.md5(json.dumps({k: getattr(args, k) for k in keys}, sort_keys=True).encode()).hexdigest()[:10]
    path = os.path.join(args.out_dir, "data_processed", f"prep_{h}.pkl")
    if os.path.exists(path):
        log.info(f"loading cached preprocessing {path}")
        with open(path, "rb") as f:
            return pickle.load(f)

    t0 = time.time()
    ds = load_yelp(args.cache_dir)
    full_labels = np.array(ds["train"]["label"])
    eda = {"full_train_size": int(len(full_labels)), "full_test_size": int(len(ds["test"])),
           "full_train_class_counts": {int(k): int(v) for k, v in zip(*np.unique(full_labels, return_counts=True))}}

    rng = np.random.default_rng(args.seed)
    sel = rng.choice(len(full_labels), size=min(args.n_train, len(full_labels)), replace=False)
    tr = ds["train"].select(sel.tolist())
    tr_text, tr_y = list(tr["text"]), np.array(tr["label"])
    te_text, te_y = list(ds["test"]["text"]), np.array(ds["test"]["label"])

    # malformed / missing / duplicates
    bad = [i for i, t in enumerate(tr_text) if not isinstance(t, str) or not t.strip()]
    seen, dup = set(), []
    for i, t in enumerate(tr_text):
        if isinstance(t, str):
            if t in seen:
                dup.append(i)
            seen.add(t)
    drop = set(bad) | set(dup)
    eda.update({"train_missing_or_empty": len(bad), "train_exact_duplicates": len(dup),
                "test_missing_or_empty": sum(1 for t in te_text if not isinstance(t, str) or not t.strip())})
    keep = [i for i in range(len(tr_text)) if i not in drop]
    tr_text, tr_y = [tr_text[i] for i in keep], tr_y[keep]
    te_text = [t if isinstance(t, str) else "" for t in te_text]

    # raw length EDA
    tr_len = np.array([len(t.split()) for t in tr_text])
    te_len = np.array([len(t.split()) for t in te_text])
    eda["train_subsample_size"] = len(tr_text)
    eda["train_subsample_class_counts"] = {int(k): int(v) for k, v in zip(*np.unique(tr_y, return_counts=True))}
    for c in (0, 1):
        L = tr_len[tr_y == c]
        eda[f"words_class{c}"] = {"mean": float(L.mean()), "median": float(np.median(L)),
                                  "p95": float(np.percentile(L, 95)), "max": int(L.max())}

    clean = make_cleaner(args)
    tr_tok = [clean(t) for t in tr_text]
    te_tok = [clean(t) for t in te_text]
    empty_after = [i for i, t in enumerate(tr_tok) if not t]
    eda["train_empty_after_cleaning_dropped"] = len(empty_after)
    ok = [i for i, t in enumerate(tr_tok) if t]
    tr_tok, tr_y, tr_text, tr_len = [tr_tok[i] for i in ok], tr_y[ok], [tr_text[i] for i in ok], tr_len[ok]
    clean_len = np.array([len(t) for t in tr_tok])
    eda["tokens_after_cleaning"] = {"mean": float(clean_len.mean()), "median": float(np.median(clean_len)),
                                    "p95": float(np.percentile(clean_len, 95)),
                                    "pct_truncated_at_max_len": float((clean_len > args.max_len).mean())}

    # train/val split
    perm = rng.permutation(len(tr_tok))
    n_val = int(len(perm) * args.val_frac)
    va_i, tr_i = perm[:n_val], perm[n_val:]
    cnt = Counter(w for i in tr_i for w in tr_tok[i])
    itos = ["<pad>", "<unk>"] + [w for w, c in cnt.most_common(args.max_vocab - 2) if c >= args.min_freq]
    stoi = {w: i for i, w in enumerate(itos)}
    eda["vocab_size"] = len(itos)
    eda["train_token_coverage"] = float(sum(c for w, c in cnt.items() if w in stoi) / sum(cnt.values()))

    Xtr, Ltr = encode([tr_tok[i] for i in tr_i], stoi, args.max_len)
    Xva, Lva = encode([tr_tok[i] for i in va_i], stoi, args.max_len)
    Xte, Lte = encode(te_tok, stoi, args.max_len)
    q1, q2 = np.quantile(tr_len, [1 / 3, 2 / 3])
    data = {"Xtr": Xtr, "Ltr": Ltr, "ytr": tr_y[tr_i], "Xva": Xva, "Lva": Lva, "yva": tr_y[va_i],
            "Xte": Xte, "Lte": Lte, "yte": te_y, "te_text": te_text, "te_rawlen": te_len,
            "len_q": (float(q1), float(q2)), "itos": itos, "eda": eda}
    eda["test_size"] = len(te_text)
    eda["splits"] = {"train": len(tr_i), "val": len(va_i), "test": len(te_text)}
    eda["preprocessing_seconds"] = time.time() - t0
    with open(path, "wb") as f:
        pickle.dump(data, f)

    # EDA outputs
    edir = os.path.join(args.out_dir, "outputs", "eda")
    os.makedirs(edir, exist_ok=True)
    with open(os.path.join(edir, "eda.json"), "w") as f:
        json.dump(eda, f, indent=2)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for c, name in ((0, "negative"), (1, "positive")):
        ax[0].hist(np.clip(tr_len[tr_y == c], 0, 1000), bins=60, alpha=0.6, label=name)
    ax[0].set_xlabel("review length (words, clipped at 1000)"); ax[0].set_ylabel("count"); ax[0].legend()
    ax[0].set_title("Review length by class (train subsample)")
    cc = eda["train_subsample_class_counts"]
    ax[1].bar(["negative", "positive"], [cc.get(0, 0), cc.get(1, 0)])
    ax[1].set_title("Class balance (train subsample)")
    fig.tight_layout(); fig.savefig(os.path.join(edir, "eda.png"), dpi=150); plt.close(fig)
    log.info("EDA: " + json.dumps(eda))
    return data


# ----------------------------------------------------------------------------
# models (all embeddings trained from scratch)
# ----------------------------------------------------------------------------
class BoWMean(nn.Module):
    def __init__(self, V, a):
        super().__init__()
        self.emb = nn.Embedding(V, a.emb_dim, padding_idx=0)
        self.head = nn.Sequential(nn.Dropout(a.dropout), nn.Linear(a.emb_dim, a.hidden), nn.ReLU(),
                                  nn.Dropout(a.dropout), nn.Linear(a.hidden, 1))

    def forward(self, x, lens):
        e = self.emb(x)
        m = (x != 0).unsqueeze(-1).float()
        pooled = (e * m).sum(1) / m.sum(1).clamp(min=1)
        return self.head(pooled).squeeze(-1)


class TextCNN(nn.Module):
    def __init__(self, V, a):
        super().__init__()
        self.emb = nn.Embedding(V, a.emb_dim, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(a.emb_dim, a.n_filters, k) for k in a.kernels])
        self.drop = nn.Dropout(a.dropout)
        self.fc = nn.Linear(a.n_filters * len(a.kernels), 1)

    def forward(self, x, lens):
        e = self.emb(x).transpose(1, 2)
        pooled = [torch.relu(c(e)).max(dim=2).values for c in self.convs]
        return self.fc(self.drop(torch.cat(pooled, dim=1))).squeeze(-1)


class BiRNN(nn.Module):
    def __init__(self, V, a, cell="lstm"):
        super().__init__()
        self.emb = nn.Embedding(V, a.emb_dim, padding_idx=0)
        rnn = nn.LSTM if cell == "lstm" else nn.GRU
        self.rnn = rnn(a.emb_dim, a.hidden, num_layers=a.num_layers, batch_first=True, bidirectional=True,
                       dropout=a.dropout if a.num_layers > 1 else 0.0)
        self.cell, self.pooling = cell, a.pooling
        self.drop = nn.Dropout(a.dropout)
        self.fc = nn.Linear(2 * a.hidden, 1)

    def forward(self, x, lens):
        e = self.drop(self.emb(x))
        packed = nn.utils.rnn.pack_padded_sequence(e, lens.clamp(min=1).cpu(), batch_first=True,
                                                   enforce_sorted=False)
        out, h = self.rnn(packed)
        if self.pooling == "last":
            h = h[0] if self.cell == "lstm" else h
            feat = torch.cat([h[-2], h[-1]], dim=1)
        else:
            out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=x.size(1))
            mask = (torch.arange(x.size(1), device=x.device)[None, :] < lens[:, None]).unsqueeze(-1)
            feat = out.masked_fill(~mask, -1e4).max(dim=1).values
        return self.fc(self.drop(feat)).squeeze(-1)


def build_model(V, a):
    if a.model == "bow":
        return BoWMean(V, a)
    if a.model == "cnn":
        return TextCNN(V, a)
    return BiRNN(V, a, "lstm" if a.model == "bilstm" else "gru")


# ----------------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------------
def conf_counts(y, p):
    tp = int(((p == 1) & (y == 1)).sum()); tn = int(((p == 0) & (y == 0)).sum())
    fp = int(((p == 1) & (y == 0)).sum()); fn = int(((p == 0) & (y == 1)).sum())
    return tp, tn, fp, fn


def macro_f1_fast(y, p):
    tp, tn, fp, fn = conf_counts(y, p)
    f1p = 2 * tp / max(2 * tp + fp + fn, 1)
    f1n = 2 * tn / max(2 * tn + fn + fp, 1)
    return (f1p + f1n) / 2


def mcc_fast(y, p):
    tp, tn, fp, fn = conf_counts(y, p)
    den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return (tp * tn - fp * fn) / den if den > 0 else 0.0


def ece(y, prob, n_bins=15):
    pred = (prob >= 0.5).astype(int)
    conf = np.where(pred == 1, prob, 1 - prob)
    correct = (pred == y).astype(float)
    bins = np.linspace(0.5, 1.0, n_bins + 1)
    total = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0.5 else (conf >= lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def bootstrap_ci(y, p, n_boot, seed):
    rng = np.random.default_rng(seed)
    n = len(y)
    acc, mf1, mcc = [], [], []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        yy, pp = y[i], p[i]
        acc.append((yy == pp).mean()); mf1.append(macro_f1_fast(yy, pp)); mcc.append(mcc_fast(yy, pp))
    ci = lambda v: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    return {"accuracy": ci(acc), "macro_f1": ci(mf1), "mcc": ci(mcc)}


def mcnemar(y, p_base, p_exp):
    from scipy.stats import binomtest, chi2
    base_ok, exp_ok = p_base == y, p_exp == y
    b = int((base_ok & ~exp_ok).sum())
    c = int((~base_ok & exp_ok).sum())
    p_exact = binomtest(min(b, c), b + c, 0.5).pvalue if b + c > 0 else 1.0
    stat = (abs(b - c) - 1) ** 2 / (b + c) if b + c > 0 else 0.0
    return {"b_base_right_exp_wrong": b, "c_base_wrong_exp_right": c,
            "chi2_cc": stat, "p_chi2_cc": float(chi2.sf(stat, 1)), "p_exact": float(p_exact)}


def all_metrics(y, prob, thr, n_boot, seed):
    from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                                 confusion_matrix, matthews_corrcoef, precision_recall_fscore_support,
                                 roc_auc_score)
    pred = (prob >= thr).astype(int)
    m = {"accuracy": accuracy_score(y, pred)}
    for avg in ("macro", "micro", "weighted"):
        pr, rc, f1, _ = precision_recall_fscore_support(y, pred, average=avg, zero_division=0)
        m[f"precision_{avg}"], m[f"recall_{avg}"], m[f"f1_{avg}"] = pr, rc, f1
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    m.update({"cm_tn": int(tn), "cm_fp": int(fp), "cm_fn": int(fn), "cm_tp": int(tp)})
    m["roc_auc"] = roc_auc_score(y, prob)
    m["pr_auc"] = average_precision_score(y, prob)
    m["mcc"] = matthews_corrcoef(y, pred)
    m["brier"] = brier_score_loss(y, prob)
    m["ece_15bin"] = ece(y, prob)
    ci = bootstrap_ci(y, pred, n_boot, seed)
    for k, (lo, hi) in ci.items():
        m[f"{k}_ci95_low"], m[f"{k}_ci95_high"] = lo, hi
    return m, pred


def slice_table(y, pred, rawlen, texts, len_q):
    q1, q2 = len_q
    neg = np.array([bool(NEG_RE.search(t.lower())) for t in texts])
    slices = {
        f"short (<= {q1:.0f} words)": rawlen <= q1,
        f"medium ({q1:.0f}-{q2:.0f} words)": (rawlen > q1) & (rawlen <= q2),
        f"long (> {q2:.0f} words)": rawlen > q2,
        "contains negation": neg,
        "no negation": ~neg,
    }
    rows = []
    for name, m in slices.items():
        if m.sum() == 0:
            continue
        rows.append({"slice": name, "n": int(m.sum()), "macro_f1": macro_f1_fast(y[m], pred[m]),
                     "error_rate": float((y[m] != pred[m]).mean())})
    return rows, slices


def plots(y, prob, pred, out, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve
    fig, ax = plt.subplots(1, 4, figsize=(20, 4.5))
    cm = confusion_matrix(y, pred, labels=[0, 1])
    ax[0].imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax[0].text(j, i, f"{cm[i, j]:,}", ha="center", va="center")
    ax[0].set_xticks([0, 1], ["neg", "pos"]); ax[0].set_yticks([0, 1], ["neg", "pos"])
    ax[0].set_xlabel("predicted"); ax[0].set_ylabel("true"); ax[0].set_title("Confusion matrix")
    fpr, tpr, _ = roc_curve(y, prob)
    ax[1].plot(fpr, tpr); ax[1].plot([0, 1], [0, 1], "k--", lw=0.8); ax[1].set_title("ROC")
    ax[1].set_xlabel("FPR"); ax[1].set_ylabel("TPR")
    pr, rc, _ = precision_recall_curve(y, prob)
    ax[2].plot(rc, pr); ax[2].set_title("Precision-recall"); ax[2].set_xlabel("recall"); ax[2].set_ylabel("precision")
    bins = np.linspace(0, 1, 11); mids, accs = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (prob >= lo) & (prob < hi if hi < 1 else prob <= hi)
        if m.any():
            mids.append(prob[m].mean()); accs.append(y[m].mean())
    ax[3].plot([0, 1], [0, 1], "k--", lw=0.8); ax[3].plot(mids, accs, "o-")
    ax[3].set_title("Reliability (P(pos))"); ax[3].set_xlabel("mean predicted"); ax[3].set_ylabel("observed positive rate")
    fig.suptitle(title); fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def error_review(y, prob, pred, texts, slices, slice_rows, seed, path):
    idx = np.arange(len(y))
    err = pred != y
    fp = idx[(pred == 1) & (y == 0)]; fp = fp[np.argsort(-prob[fp])][:5]
    fn = idx[(pred == 0) & (y == 1)]; fn = fn[np.argsort(prob[fn])][:5]
    used = set(fp) | set(fn)
    near = [i for i in idx[err][np.argsort(np.abs(prob[err] - 0.5))] if i not in used][:5]
    used |= set(near)
    worst = max(slice_rows, key=lambda r: r["error_rate"])["slice"]
    cand = [i for i in idx[err & slices[worst]] if i not in used]
    rng = np.random.default_rng(seed)
    sl = rng.choice(cand, size=min(5, len(cand)), replace=False) if cand else []
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["category", "test_index", "true_label", "p_positive", "text", "error_type", "notes_and_fix"])
        for cat, ids in (("confident_false_positive", fp), ("confident_false_negative", fn),
                         ("near_threshold", near), (f"slice: {worst}", sl)):
            for i in ids:
                w.writerow([cat, int(i), int(y[i]), f"{prob[i]:.4f}", texts[i][:1200].replace("\\n", " "), "", ""])


def merge_metrics(path, run, rows):
    old = []
    if os.path.exists(path):
        with open(path) as f:
            old = [r for r in csv.DictReader(f) if r["model"] != run]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["model", "metric", "value"])
        w.writeheader()
        for r in old:
            w.writerow(r)
        for k, v in rows:
            w.writerow({"model": run, "metric": k, "value": f"{v:.6g}" if isinstance(v, float) else v})


# ----------------------------------------------------------------------------
# train / predict
# ----------------------------------------------------------------------------
@torch.no_grad()
def predict(model, X, L, bs, device):
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        x = torch.from_numpy(X[i:i + bs]).to(device); l = torch.from_numpy(L[i:i + bs]).to(device)
        out.append(torch.sigmoid(model(x, l)).float().cpu().numpy())
    return np.concatenate(out)


def train_mode(args, log):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = prepare_data(args, log)
    V = len(data["itos"])
    model = build_model(V, args).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"model={args.model} params={n_params:,} vocab={V}")
    ds = TensorDataset(torch.from_numpy(data["Xtr"]), torch.from_numpy(data["Ltr"]),
                       torch.from_numpy(data["ytr"]).float())
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                    pin_memory=device == "cuda", drop_last=False)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * len(dl),
                                                pct_start=0.1)
    loss_fn = nn.BCEWithLogitsLoss()
    run_dir = os.path.join(args.out_dir, "outputs", args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    ck_path = os.path.join(args.out_dir, "checkpoints", f"{args.run_name}_best.pt")
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    best, bad_epochs, train_time, n_seen, hist = float("inf"), 0, 0.0, 0, []
    for ep in range(1, args.epochs + 1):
        model.train()
        t0 = time.time(); tot, nb = 0.0, 0
        for x, l, y in dl:
            x, l, y = x.to(device, non_blocking=True), l.to(device), y.to(device)
            loss = loss_fn(model(x, l), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step(); sched.step()
            tot += loss.item() * len(y); nb += len(y)
        if device == "cuda":
            torch.cuda.synchronize()
        train_time += time.time() - t0; n_seen += nb
        pv = predict(model, data["Xva"], data["Lva"], 1024, device)
        yv = data["yva"]
        vloss = float(-np.mean(yv * np.log(np.clip(pv, 1e-7, 1)) + (1 - yv) * np.log(np.clip(1 - pv, 1e-7, 1))))
        vacc = float(((pv >= 0.5) == yv).mean())
        hist.append({"epoch": ep, "train_loss": tot / nb, "val_loss": vloss, "val_acc": vacc})
        log.info(f"epoch {ep} train_loss {tot / nb:.4f} val_loss {vloss:.4f} val_acc {vacc:.4f} "
                 f"({time.time() - t0:.0f}s)")
        if vloss < best - 1e-4:
            best, bad_epochs = vloss, 0
            torch.save({"model": model.state_dict(), "args": vars(args), "itos": data["itos"], "epoch": ep}, ck_path)
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                log.info("early stopping")
                break
    peak_gb = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else float("nan")
    model.load_state_dict(torch.load(ck_path, map_location=device)["model"])

    yte = data["yte"]
    prob = predict(model, data["Xte"], data["Lte"], 1024, device)
    m, pred = all_metrics(yte, prob, args.threshold, args.n_boot, args.seed)
    srows, slices = slice_table(yte, pred, data["te_rawlen"], data["te_text"], data["len_q"])
    np.savez(os.path.join(run_dir, "test_preds.npz"), y=yte, prob=prob, pred=pred)
    plots(yte, prob, pred, os.path.join(run_dir, "eval_plots.png"), f"{args.run_name} ({args.model})")
    error_review(yte, prob, pred, data["te_text"], slices, srows, args.seed,
                 os.path.join(run_dir, "error_review.csv"))
    with open(os.path.join(run_dir, "slices.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["slice", "n", "macro_f1", "error_rate"]); w.writeheader(); w.writerows(srows)
    with open(os.path.join(run_dir, "history.json"), "w") as f:
        json.dump(hist, f, indent=2)

    rows = [("model_type", args.model)] + list(m.items())
    for r in srows:
        rows += [(f"slice_macro_f1[{r['slice']}]", r["macro_f1"]), (f"slice_error_rate[{r['slice']}]", r["error_rate"])]
    rows += [("parameter_count", n_params), ("training_time_s", train_time),
             ("train_examples_per_sec", n_seen / max(train_time, 1e-9)), ("peak_gpu_memory_gb", peak_gb),
             ("epochs_run", len(hist)), ("device", device_name()),
             ("hyperparameters", json.dumps({k: getattr(args, k) for k in (
                 "model", "emb_dim", "hidden", "num_layers", "kernels", "n_filters", "pooling", "dropout",
                 "lr", "batch_size", "weight_decay", "max_len", "epochs")}))]
    merge_metrics(os.path.join(args.out_dir, "metrics_report.csv"), args.run_name, rows)
    log.info("test metrics: " + json.dumps(m))
    log.info("slices: " + json.dumps(srows))
    log.info(f"wrote {run_dir}/ and metrics_report.csv")


def compare_mode(args, log):
    base = np.load(args.baseline_preds)
    rows, out = [], os.path.join(args.out_dir, "metrics_report.csv")
    for p in args.exp_preds:
        e = np.load(p)
        assert np.array_equal(base["y"], e["y"]), "test sets differ; were both runs on the full test split?"
        r = mcnemar(base["y"], base["pred"], e["pred"])
        name = os.path.basename(os.path.dirname(p))
        log.info(f"McNemar baseline vs {name}: {r}")
        rows += [(f"mcnemar_vs_{name}_{k}", v) for k, v in r.items()]
    merge_metrics(out, "mcnemar_" + os.path.basename(os.path.dirname(args.baseline_preds)), rows)
    # side-by-side table of all runs for the report
    with open(out) as f:
        recs = list(csv.DictReader(f))
    models = sorted({r["model"] for r in recs if not r["model"].startswith("mcnemar")})
    metrics = []
    for r in recs:
        if r["model"] in models and r["metric"] not in metrics:
            metrics.append(r["metric"])
    table = {(r["model"], r["metric"]): r["value"] for r in recs}
    with open(os.path.join(args.out_dir, "outputs", "comparison_table.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["metric"] + models)
        for mt in metrics:
            w.writerow([mt] + [table.get((m, mt), "") for m in models])
    log.info("wrote outputs/comparison_table.csv")


def main():
    args = parse_args()
    set_seed(args.seed)
    log = setup_run(args)
    if args.mode == "train":
        train_mode(args, log)
    else:
        compare_mode(args, log)


if __name__ == "__main__":
    main()

