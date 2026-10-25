#!/usr/bin/env python
"""E12 (MPNet bi-encoder + geometric features + frozen NLI score), fixed and
parallelised for Kaggle "GPU T4 x2".

Run it from a notebook cell as a script (multiprocessing needs a real __main__):
    !python /kaggle/working/ambistory/kaggle/e12_train.py

Parallelism: one worker process per GPU, pulling jobs from a shared queue.
  * phase 1: the frozen NLI cross-encoder scores are sharded across GPUs;
  * phase 2: CV folds (x seeds) are independent jobs, so each GPU trains its own
    folds. 5 folds on 2 GPUs = 3 rounds instead of 5; use --seeds with 2 values
    to get 10 jobs and keep both GPUs busy the whole time.
  Fold-level parallelism beats DDP here: the model is small, DDP would sync
  gradients every step and would split story-grouped batches across devices.

Speed-ups vs. the notebook: fp16 autocast on the T4 tensor cores, de-duplicated
texts (each story text occurs 2x and each meaning 3x, so they are encoded once),
length-sorted inference batches, vectorised losses and group features,
sklearn-free PCA, cached NLI scores, and resumable folds (finished folds are
skipped on re-run, useful against Kaggle session limits).

Fixes from REVIEW.md: no per-homonym mean (C3); MLP epoch/SWA selection on an
inner split, never the outer fold (C4); story-grouped batches so the ranking
losses actually see pairs (H1); explicit ending flag (M8); soft-rank blend kept
only if it improves OOF (M6); official metric and JSONL output (M5).
Still open: H5 (the MLP is trained on features from rows the encoder saw).
"""
import argparse
import glob
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from ambistory import data as D  # noqa: E402
from ambistory import metrics as M  # noqa: E402
from ambistory import submission as S  # noqa: E402
from ambistory.batching import RandomBatchSampler, StoryBatchSampler  # noqa: E402
from ambistory.losses import within_setup_ranking_loss  # noqa: E402

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=None, help="dir with train/dev/test.json (default: search /kaggle/input)")
    p.add_argument("--out", default="/kaggle/working/e12" if os.path.isdir("/kaggle/working") else "outputs/e12")
    p.add_argument("--mpnet", default="sentence-transformers/all-mpnet-base-v2", help="HF id or local dir")
    p.add_argument("--ce", default="cross-encoder/nli-deberta-v3-base", help="HF id or local dir")
    p.add_argument("--gpus", default="auto", help="'auto', 'cpu', or e.g. '0,1'")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seeds", default="42", help="comma-separated; e.g. '42,43' for 10 jobs on 2 GPUs")
    # encoder fine-tuning
    p.add_argument("--ft_epochs", type=int, default=6)
    p.add_argument("--ft_lr", type=float, default=2e-5)
    p.add_argument("--ft_stories_per_batch", type=int, default=4, help="x6 rows per batch")
    p.add_argument("--ft_rank_w", type=float, default=0.5)
    p.add_argument("--ft_warmup", type=float, default=0.1)
    p.add_argument("--ft_patience", type=int, default=2)
    p.add_argument("--max_len", type=int, default=128)
    p.add_argument("--ce_max_len", type=int, default=256)
    p.add_argument("--inner_val_frac", type=float, default=0.15)
    # features + MLP
    p.add_argument("--pca", type=int, default=96)
    p.add_argument("--mlp_h1", type=int, default=512)
    p.add_argument("--mlp_h2", type=int, default=128)
    p.add_argument("--mlp_dropout", type=float, default=0.3)
    p.add_argument("--mlp_lr", type=float, default=8e-4)
    p.add_argument("--mlp_wd", type=float, default=3e-4)
    p.add_argument("--mlp_epochs", type=int, default=60)
    p.add_argument("--mlp_stories_per_batch", type=int, default=5)
    p.add_argument("--mlp_rank_w", type=float, default=0.3)
    p.add_argument("--mlp_rank_margin", type=float, default=0.5)
    p.add_argument("--mlp_mse_w", type=float, default=0.2)
    p.add_argument("--swa_start", type=float, default=0.75)
    p.add_argument("--rank_blend_w", type=float, default=0.15)
    p.add_argument("--save_models", action="store_true", help="save fp16 encoder + MLP per fold (~220 MB each)")
    p.add_argument("--fresh", action="store_true", help="ignore cached CE scores / finished folds")
    p.add_argument("--limit_stories", type=int, default=0, help="smoke test: keep only N stories per split")
    # ablation switches (defaults = the fixed E12 system); see kaggle/ablations.py
    p.add_argument("--hom_mean", action="store_true", help="add the per-homonym label mean feature (C3)")
    p.add_argument("--nons_meta", action="store_true",
                   help="add Model 7-9's 'nonsensical' meta feature, computed the original (buggy) way (C1)")
    p.add_argument("--random_batches", action="store_true", help="shuffled instead of story-grouped batches (H1)")
    p.add_argument("--no_ce", action="store_true", help="drop the frozen NLI entailment features")
    p.add_argument("--select_on_outer", action="store_true",
                   help="train the MLP on all fold-train rows and pick its epoch on the scored fold (C4)")
    p.add_argument("--test_gold", default=None,
                   help="labelled test.json, used ONLY to report the final test score (never for training/selection)")
    p.add_argument("--ce_label_index", type=int, default=-1, help="override entailment index (-1 = from config)")
    return p.parse_args(argv)


def find_data_dir(given):
    if given:
        return given
    for path in sorted(glob.glob("/kaggle/input/**/train.json", recursive=True)):
        d = os.path.dirname(path)
        if all(os.path.exists(os.path.join(d, f)) for f in ("dev.json", "test.json")):
            return d
    raise FileNotFoundError("no dir with train/dev/test.json under /kaggle/input; pass --data")


def resolve_model(name):
    """Download once in the parent so workers don't race on the HF cache."""
    if os.path.isdir(name):
        return name
    from huggingface_hub import snapshot_download
    path = snapshot_download(name, allow_patterns=["*.json", "*.txt", "*.model", "*.safetensors"])
    if not glob.glob(os.path.join(path, "*.safetensors")):  # older repos only ship .bin weights
        path = snapshot_download(name, allow_patterns=["*.json", "*.txt", "*.model", "*.bin"])
    return path


# ---------------------------------------------------------------------------
# Data (deterministic, recomputed in every process instead of pickled)
# ---------------------------------------------------------------------------
def _limit(rows, n):
    if not n:
        return rows
    keep = list(dict.fromkeys(D.story_id(r) for r in rows))[:n]
    keep = set(keep)
    return [r for r in rows if D.story_id(r) in keep]


def annotator_dist(row):
    """5-class soft target from the votes that were not flagged nonsensical."""
    votes = [c for c, ns in zip(row["choices"], row["nonsensical"]) if not ns] or row["choices"]
    dist = np.bincount(np.asarray(votes, dtype=int) - 1, minlength=5).astype(np.float32)
    return dist / dist.sum()


def prepare(cfg):
    d = cfg.data_dir
    traindev = _limit(D.load_split(os.path.join(d, "train.json")), cfg.limit_stories) + \
        _limit(D.load_split(os.path.join(d, "dev.json")), cfg.limit_stories)
    test = _limit(D.load_split(os.path.join(d, "test.json")), cfg.limit_stories)
    rows = traindev + test
    sids = {s: i for i, s in enumerate(dict.fromkeys(D.setup_id(r) for r in rows))}
    stids = {s: i for i, s in enumerate(dict.fromkeys(D.story_id(r) for r in rows))}
    T = {
        "story": [D.story_text(r) for r in rows],
        "meaning": [D.meaning_text(r) for r in rows],
        "ending": [r["ending"] or r["sentence"].strip() for r in rows],  # sense-resolving fragment
        "ambiguous": [r["sentence"].strip() for r in rows],
        "gloss": [r["judged_meaning"].strip() for r in rows],
        "setup": np.array([sids[D.setup_id(r)] for r in rows]),
        "story_gid": np.array([stids[D.story_id(r)] for r in rows]),
        "has_ending": np.array([D.ending_condition(r) for r in rows], dtype=np.float32),
    }
    n_td = len(traindev)
    T["y"] = np.array([r["average"] for r in traindev], dtype=np.float32)
    T["sd"] = np.array([r["stdev"] for r in traindev], dtype=np.float32)
    T["dist"] = np.stack([annotator_dist(r) for r in traindev])
    folds = np.array(D.homonym_folds(traindev, cfg.folds))
    if getattr(cfg, "nons_meta", False):  # reproduce the bug: test holds the string "(???)" -> 5 truthy chars
        raw = {}
        for split in ("train", "dev", "test"):
            with open(os.path.join(d, f"{split}.json"), encoding="utf8") as f:
                raw[split] = json.load(f)
        T["nons"] = np.array([min(sum(1 for v in raw[r["split"]][r["id"]].get("nonsensical", []) if v) / 5.0, 1.0)
                              for r in rows], dtype=np.float32)
    return traindev, test, T, n_td, folds


def inner_split(rows, idx, frac, seed):
    """Homonym-grouped hold-out inside the fold-train rows (for early stopping only)."""
    rng = np.random.default_rng(seed)
    homs = sorted({rows[i]["homonym"] for i in idx})
    rng.shuffle(homs)
    target, held, n = frac * len(idx), set(), 0
    per_hom = {}
    for i in idx:
        per_hom[rows[i]["homonym"]] = per_hom.get(rows[i]["homonym"], 0) + 1
    for h in homs:
        if n >= target:
            break
        held.add(h)
        n += per_hom[h]
    va = np.array([i for i in idx if rows[i]["homonym"] in held])
    tr = np.array([i for i in idx if rows[i]["homonym"] not in held])
    return tr, va


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class SentenceEncoder(nn.Module):
    def __init__(self, path, max_len):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModel.from_pretrained(path).float()  # fp32 master weights (see e13_train.py)
        self.max_len = max_len

    def forward(self, texts):
        dev = next(self.model.parameters()).device
        enc = self.tok(texts, padding=True, truncation=True, max_length=self.max_len,
                       return_tensors="pt").to(dev)
        hidden = self.model(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        emb = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return F.normalize(emb.float(), dim=-1)

    def encode_unique(self, texts):
        """Encode each distinct text once; gradients flow back through the gather."""
        pos = {}
        for t in texts:
            pos.setdefault(t, len(pos))
        idx = torch.tensor([pos[t] for t in texts], device=next(self.model.parameters()).device)
        return self(list(pos))[idx]


@torch.no_grad()
def embed_all(enc, texts, batch=128):
    """Inference over distinct texts, length-sorted to minimise padding. Returns np [N, d]."""
    enc.eval()
    uniq = list(dict.fromkeys(texts))
    order = sorted(range(len(uniq)), key=lambda i: len(uniq[i]))
    out = [None] * len(uniq)
    for s in range(0, len(order), batch):
        ids = order[s:s + batch]
        with torch.autocast("cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
            e = enc([uniq[i] for i in ids]).cpu().numpy()
        for j, i in enumerate(ids):
            out[i] = e[j]
    pos = {t: i for i, t in enumerate(uniq)}
    return np.stack([out[pos[t]] for t in texts]).astype(np.float32)


class OrdinalHead(nn.Module):
    """Proportional-odds head: 5 ordered classes from one logit and 4 monotone thresholds."""

    def __init__(self, in_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, 1, bias=False)
        self.raw_thresholds = nn.Parameter(torch.linspace(-2.0, 2.0, 4))

    def forward(self, h):
        thresholds = torch.cumsum(F.softplus(self.raw_thresholds), dim=0)
        cum = torch.sigmoid(thresholds - self.linear(h))
        cum = torch.cat([torch.zeros_like(cum[:, :1]), cum, torch.ones_like(cum[:, :1])], dim=1)
        probs = (cum[:, 1:] - cum[:, :-1]).clamp(min=1e-9)
        return probs, (probs * torch.arange(1, 6, device=h.device, dtype=h.dtype)).sum(-1)


class WSDModel(nn.Module):
    def __init__(self, in_dim, h1, h2, drop):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, h1), nn.LayerNorm(h1), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(h1, h2), nn.LayerNorm(h2), nn.ReLU(), nn.Dropout(drop))
        self.head = OrdinalHead(h2)

    def forward(self, x):
        return self.head(self.backbone(x))


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
def group_mean(x, gid):
    s = np.bincount(gid, weights=x)
    c = np.bincount(gid)
    return (s / np.maximum(c, 1))[gid]


def group_rank(x, gid):
    """Rank of x within its group, scaled to [0, 1] (0/1 for the 2-row setups)."""
    order = np.lexsort((x, gid))
    g_sorted = gid[order]
    starts = np.r_[0, np.flatnonzero(np.diff(g_sorted)) + 1]
    sizes = np.diff(np.r_[starts, len(order)])
    first = np.repeat(starts, sizes)
    rank = np.empty(len(x), dtype=np.float32)
    rank[order] = (np.arange(len(order)) - first) / np.maximum(np.repeat(sizes, sizes) - 1, 1)
    return rank


class PCA:
    def fit(self, X, k):
        self.mu = X.mean(0)
        _, _, vt = np.linalg.svd(X - self.mu, full_matrices=False)
        self.components = vt[:k]
        return self

    def transform(self, X):
        return ((X - self.mu) @ self.components.T).astype(np.float32)


def raw_features(E, T, idx, ce):
    hs, hm, he, ha = (E[k][idx] for k in ("story", "meaning", "ending", "ambiguous"))
    gid = T["setup"][idx]
    cos = (hs * hm).sum(1)
    c = ce[idx]
    st = [T["story"][i] for i in idx]
    mn = [T["meaning"][i] for i in idx]
    tok = np.array([len(set(a.lower().split()) & set(b.lower().split())) /
                    max(len(set(a.lower().split()) | set(b.lower().split())), 1) for a, b in zip(st, mn)])
    lenr = np.array([len(a.split()) / max(len(b.split()), 1) for a, b in zip(st, mn)])
    scalars = np.stack([cos, cos - group_mean(cos, gid), c, group_rank(cos, gid), c - group_mean(c, gid),
                        (he * hm).sum(1), (ha * hm).sum(1), tok, lenr, T["has_ending"][idx]], 1)
    return scalars.astype(np.float32), np.abs(hs - hm), hs * hm


def build_features(E, T, ce, fit_idx, all_idx, k):
    sc, l1, had = raw_features(E, T, fit_idx, ce)
    p1, p2 = PCA().fit(l1, k), PCA().fit(had, k)
    sc, l1, had = raw_features(E, T, all_idx, ce)
    return np.concatenate([sc, p1.transform(l1), p2.transform(had)], 1), (p1, p2)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def cosine_warmup(opt, warmup, total):
    def f(step):
        if step < warmup:
            return step / max(1, warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warmup) / max(1, total - warmup))))
    return torch.optim.lr_scheduler.LambdaLR(opt, f)


def finetune_encoder(enc, T, tr, va, cfg, seed, log):
    dev = next(enc.parameters()).device
    amp = dev.type == "cuda"
    opt = torch.optim.AdamW(enc.parameters(), lr=cfg.ft_lr, weight_decay=1e-2)
    sampler = RandomBatchSampler(len(tr), 6 * cfg.ft_stories_per_batch, seed=seed) if cfg.random_batches         else StoryBatchSampler(T["story_gid"][tr], cfg.ft_stories_per_batch, seed=seed)
    total = cfg.ft_epochs * len(sampler)
    sched = cosine_warmup(opt, int(cfg.ft_warmup * total), total)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    y = torch.tensor((T["y"][tr] - 1) / 4, device=dev)
    gid = torch.tensor(T["setup"][tr], device=dev)
    stories = [T["story"][i] for i in tr]
    meanings = [T["meaning"][i] for i in tr]
    va_story, va_meaning = [T["story"][i] for i in va], [T["meaning"][i] for i in va]
    best, best_state, bad = -2.0, None, 0
    for ep in range(1, cfg.ft_epochs + 1):
        enc.train()
        t0, n_pairs = time.time(), 0
        for b in sampler:
            bt = torch.tensor(b, device=dev)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                hs = enc.encode_unique([stories[i] for i in b])
                hm = enc.encode_unique([meanings[i] for i in b])
            cos = (hs * hm).sum(-1)
            rank, n = within_setup_ranking_loss(cos, y[bt], gid[bt], margin=0.1, min_gap=0.05)
            loss = F.mse_loss(cos, y[bt]) + cfg.ft_rank_w * rank
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(enc.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            n_pairs += n
        cos_va = (embed_all(enc, va_story) * embed_all(enc, va_meaning)).sum(1)
        sp = M.spearman(cos_va, T["y"][va])
        log(f"  FT ep {ep}/{cfg.ft_epochs} inner-val rho={sp:.4f} rank-pairs={n_pairs} ({time.time() - t0:.0f}s)")
        if sp > best:
            best, bad = sp, 0
            best_state = {k: v.detach().cpu().clone() for k, v in enc.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.ft_patience:
                log(f"  FT early stop at epoch {ep}")
                break
    if best_state is not None:  # None when --ft_epochs 0 (frozen pretrained encoder)
        enc.load_state_dict(best_state)
    return best


def train_mlp(X, T, tr, va, cfg, seed, dev, log):
    from torch.optim.swa_utils import AveragedModel, SWALR
    torch.manual_seed(seed)
    Xt = torch.tensor(X, device=dev)
    y = torch.tensor(T["y"], device=dev)
    dist = torch.tensor(0.9 * T["dist"] + 0.02, device=dev)  # label smoothing toward uniform
    gid = torch.tensor(T["setup"], device=dev)
    model = WSDModel(X.shape[1], cfg.mlp_h1, cfg.mlp_h2, cfg.mlp_dropout).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.mlp_lr, weight_decay=cfg.mlp_wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.mlp_epochs, eta_min=cfg.mlp_lr * 0.05)
    swa, swa_sched = AveragedModel(model), SWALR(opt, swa_lr=cfg.mlp_lr * 0.05)
    swa_start = max(1, int(cfg.swa_start * cfg.mlp_epochs))
    sampler = RandomBatchSampler(len(tr), 6 * cfg.mlp_stories_per_batch, seed=seed) if cfg.random_batches         else StoryBatchSampler(T["story_gid"][tr], cfg.mlp_stories_per_batch, seed=seed)
    va_t = torch.tensor(va, device=dev)
    best, best_state = -2.0, None
    for ep in range(1, cfg.mlp_epochs + 1):
        model.train()
        for b in sampler:
            bi = torch.tensor(tr[b], device=dev)
            probs, exp = model(Xt[bi])
            rank, _ = within_setup_ranking_loss(exp, y[bi], gid[bi], margin=cfg.mlp_rank_margin, min_gap=0.1)
            loss = F.kl_div(torch.log(probs), dist[bi], reduction="batchmean") \
                + cfg.mlp_rank_w * rank + cfg.mlp_mse_w * F.mse_loss(exp, y[bi])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        if ep >= swa_start:
            swa.update_parameters(model)
            swa_sched.step()
        else:
            sched.step()
        current = swa.module if ep >= swa_start else model
        current.eval()
        with torch.no_grad():
            p = current(Xt[va_t])[1].cpu().numpy()
        sp = M.spearman(p, T["y"][va])
        if sp > best:
            best = sp
            best_state = {k: v.detach().clone() for k, v in current.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    log(f"  MLP best inner-val rho={best:.4f}")
    return model, best


# ---------------------------------------------------------------------------
# Jobs (run inside worker processes)
# ---------------------------------------------------------------------------
def job_ce(shard, n_shards, ctx):
    cfg, T, dev, log = ctx["cfg"], ctx["T"], ctx["dev"], ctx["log"]
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.ce_path)
    model = AutoModelForSequenceClassification.from_pretrained(cfg.ce_path).to(dev).eval()
    ent = cfg.ce_label_index
    if ent < 0:
        labels = {v.lower(): int(k) for k, v in model.config.id2label.items()}
        ent = next(i for name, i in labels.items() if "entail" in name)
    pairs = list(dict.fromkeys(zip(T["story"], T["gloss"])))[shard::n_shards]
    order = sorted(range(len(pairs)), key=lambda i: len(pairs[i][0]))
    scores = {}
    t0 = time.time()
    with torch.no_grad():
        for s in range(0, len(order), 64):
            batch = [pairs[i] for i in order[s:s + 64]]
            enc = tok([a for a, _ in batch], [b for _, b in batch], padding=True, truncation=True,
                      max_length=cfg.ce_max_len, return_tensors="pt").to(dev)
            with torch.autocast("cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
                logits = model(**enc).logits
            p = torch.softmax(logits.float(), -1)[:, ent].cpu().numpy()
            scores.update(zip(batch, p.tolist()))
    with open(os.path.join(cfg.out, f"ce_shard{shard}.json"), "w") as f:
        json.dump([[a, b, s] for (a, b), s in scores.items()], f)
    log(f"CE shard {shard}/{n_shards}: {len(pairs)} pairs in {time.time() - t0:.0f}s (entailment idx {ent})")


def job_fold(fold, seed, ctx):
    cfg, T, dev, log, folds, n_td = ctx["cfg"], ctx["T"], ctx["dev"], ctx["log"], ctx["folds"], ctx["n_td"]
    t0 = time.time()
    torch.manual_seed(seed + fold)
    np.random.seed(seed + fold)
    outer_tr = np.flatnonzero(folds != fold)
    outer_va = np.flatnonzero(folds == fold)
    test_idx = np.arange(n_td, len(T["story"]))
    in_tr, in_va = inner_split(ctx["traindev"], outer_tr, cfg.inner_val_frac, seed * 100 + fold)
    log(f"fold {fold} seed {seed}: inner-train={len(in_tr)} inner-val={len(in_va)} "
        f"outer-val={len(outer_va)} test={len(test_idx)}")

    enc = SentenceEncoder(cfg.mpnet_path, cfg.max_len).to(dev)
    ft_best = finetune_encoder(enc, T, in_tr, in_va, cfg, seed + fold, log)

    E = {k: embed_all(enc, T[k]) for k in ("story", "meaning", "ending", "ambiguous")}
    ce = np.zeros_like(ctx["ce"]) if cfg.no_ce else ctx["ce"]
    X, pcas = build_features(E, T, ce, in_tr, np.arange(len(T["story"])), min(cfg.pca, len(in_tr) - 1))
    extra = []
    if cfg.hom_mean:  # as in the original: mean label per homonym over fold-train rows (own label included)
        rows = ctx["traindev"]
        means = {}
        for i in outer_tr:
            means.setdefault(rows[i]["homonym"], []).append(T["y"][i])
        homs = [r["homonym"] for r in rows] + [r["homonym"] for r in ctx["test"]]
        extra.append(np.array([np.mean(means[h]) if h in means else 3.0 for h in homs], dtype=np.float32))
    if cfg.nons_meta:
        extra.append(T["nons"])
    if extra:
        X = np.concatenate([X, np.stack(extra, 1)], 1).astype(np.float32)
    mlp_tr, mlp_va = (outer_tr, outer_va) if cfg.select_on_outer else (in_tr, in_va)
    mlp, mlp_best = train_mlp(X, T, mlp_tr, mlp_va, cfg, seed + fold, dev, log)
    with torch.no_grad():
        Xt = torch.tensor(X, device=dev)
        pred = np.clip(mlp(Xt)[1].cpu().numpy(), 1.0, 5.0)

    m = M.evaluate(pred[outer_va], [ctx["traindev"][i] for i in outer_va])
    log(f"fold {fold} seed {seed}: OUTER rho={m['spearman']:.4f} acc={m['accuracy']:.4f} "
        f"({(time.time() - t0) / 60:.1f} min)")
    name = os.path.join(cfg.out, f"fold{fold}_seed{seed}")
    np.savez(name + ".tmp.npz", val_idx=outer_va, val_pred=pred[outer_va], test_pred=pred[test_idx],
             ft_inner=ft_best, mlp_inner=mlp_best)
    os.replace(name + ".tmp.npz", name + ".npz")  # atomic: a half-written fold is never "done"
    if cfg.save_models:
        torch.save({"encoder": {k: v.half() for k, v in enc.state_dict().items()},
                    "mlp": mlp.state_dict(), "pca": [(p.mu, p.components) for p in pcas],
                    "config": vars(cfg)}, name + "_models.pt")
    del enc, mlp
    if dev.type == "cuda":
        torch.cuda.empty_cache()


def worker(gpu, queue, cfg):
    dev = torch.device(f"cuda:{gpu}" if gpu is not None else "cpu")
    if dev.type == "cuda":
        torch.cuda.set_device(dev)
        torch.backends.cudnn.benchmark = True
    tag = f"[{'gpu' + str(gpu) if gpu is not None else 'cpu'}]"
    log = lambda msg: print(f"{time.strftime('%H:%M:%S')} {tag} {msg}", flush=True)  # noqa: E731
    traindev, test, T, n_td, folds = prepare(cfg)
    ctx = {"cfg": cfg, "T": T, "dev": dev, "log": log, "folds": folds, "n_td": n_td, "traindev": traindev,
           "test": test}
    ce_path = os.path.join(cfg.out, "ce_scores.npy")
    if os.path.exists(ce_path):
        ctx["ce"] = np.load(ce_path)
    while True:
        job = queue.get()
        if job is None:
            break
        if job[0] == "ce":
            job_ce(job[1], job[2], ctx)
        else:
            job_fold(job[1], job[2], ctx)


def run_jobs(jobs, devices, cfg, target=None):
    """Run jobs on one worker process per device. target(gpu, queue, cfg) defaults to this file's worker."""
    if not jobs:
        return
    mpctx = mp.get_context("spawn")
    queue = mpctx.Queue()
    for j in jobs:
        queue.put(j)
    n_workers = min(len(devices), len(jobs))
    for _ in range(n_workers):
        queue.put(None)
    procs = [mpctx.Process(target=target or worker, args=(devices[i], queue, cfg)) for i in range(n_workers)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    failed = [p.exitcode for p in procs if p.exitcode != 0]
    if failed:
        raise RuntimeError(f"{len(failed)} worker(s) failed (exit codes {failed}); see the log above")


# ---------------------------------------------------------------------------
# Main (parent process: no training here)
# ---------------------------------------------------------------------------
def soft_rank_blend(pred, gid, w):
    """Pull the higher-ranked sense of each setup toward 5 and the lower toward 1."""
    return np.clip((1 - w) * pred + w * (group_rank(pred, gid) * 4 + 1), 1.0, 5.0)


def get_devices(spec):
    if spec == "cpu" or not torch.cuda.is_available():
        return [None], ["cpu"]
    devices = list(range(torch.cuda.device_count())) if spec == "auto" else [int(g) for g in spec.split(",")]
    return devices, [torch.cuda.get_device_name(g) for g in devices]


def load_gold(path, test):
    """Labelled test rows aligned with `test`, or None. Used only for final reporting."""
    if not path:
        return None
    gold = {r["id"]: r for r in D.load_split(path)}
    gold_rows = [gold[r["id"]] for r in test]
    assert all(g["judged_meaning"] == r["judged_meaning"] and g["sentence"] == r["sentence"]
               for g, r in zip(gold_rows, test)), "test_gold rows do not match test.json"
    return gold_rows if D.has_labels(gold_rows) else None


def aggregate(cfg, traindev, test, T, n_td, seeds, t0, names):
    """Combine fold files into OOF/test predictions, metrics.json and predictions.jsonl."""
    oof = np.zeros((len(seeds), n_td), dtype=np.float32)
    test_pred = np.zeros((len(seeds), len(test)), dtype=np.float32)
    per_fold = []
    for si, s in enumerate(seeds):
        for k in range(cfg.folds):
            r = np.load(os.path.join(cfg.out, f"fold{k}_seed{s}.npz"))
            oof[si, r["val_idx"]] = r["val_pred"]
            test_pred[si] += r["test_pred"] / cfg.folds
            per_fold.append(M.evaluate(r["val_pred"], [traindev[i] for i in r["val_idx"]]))
    oof, test_pred = oof.mean(0), test_pred.mean(0)
    base = M.evaluate(oof, traindev)
    blended = M.evaluate(soft_rank_blend(oof, T["setup"][:n_td], cfg.rank_blend_w), traindev)
    use_blend = blended["spearman"] > base["spearman"]
    final = soft_rank_blend(test_pred, T["setup"][n_td:], cfg.rank_blend_w) if use_blend else test_pred

    S.write_predictions([r["id"] for r in test], final, os.path.join(cfg.out, "predictions.jsonl"))
    S.check_against(os.path.join(cfg.out, "predictions.jsonl"), test)
    np.save(os.path.join(cfg.out, "oof.npy"), oof)
    np.save(os.path.join(cfg.out, "test_pred.npy"), test_pred)  # before the rank blend, for ensembling
    summary = {"oof": base, "oof_with_rank_blend": blended, "rank_blend_used": use_blend,
               "per_fold_mean_std": M.summarize_folds(per_fold), "seeds": seeds,
               "minutes": (time.time() - t0) / 60, "devices": names}
    gold_rows = load_gold(cfg.test_gold, test)
    if gold_rows:
        summary["test"] = M.evaluate(final, gold_rows)
        summary["test_without_rank_blend"] = M.evaluate(test_pred, gold_rows)
        print(f"TEST (official metric) rho={summary['test']['spearman']:.4f} "
              f"acc={summary['test']['accuracy']:.4f} | without blend "
              f"rho={summary['test_without_rank_blend']['spearman']:.4f}", flush=True)
    with open(os.path.join(cfg.out, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)
    pf = summary["per_fold_mean_std"]
    print(f"\nOOF rho={base['spearman']:.4f} acc={base['accuracy']:.4f} | "
          f"per-fold rho={pf['spearman'][0]:.4f}+-{pf['spearman'][1]:.4f} | "
          f"rank blend {'used' if use_blend else 'NOT used'} (OOF rho {blended['spearman']:.4f})")
    print(f"wrote {os.path.join(cfg.out, 'predictions.jsonl')}  ({summary['minutes']:.1f} min total)")
    return summary


def main():
    cfg = parse_args()
    cfg.data_dir = find_data_dir(cfg.data)
    os.makedirs(cfg.out, exist_ok=True)
    seeds = [int(s) for s in cfg.seeds.split(",")]
    devices, names = get_devices(cfg.gpus)
    print(f"data={cfg.data_dir}\nout={cfg.out}\ndevices={names}", flush=True)

    t0 = time.time()
    cfg.mpnet_path, cfg.ce_path = resolve_model(cfg.mpnet), resolve_model(cfg.ce)
    traindev, test, T, n_td, folds = prepare(cfg)
    with open(os.path.join(cfg.out, "folds.json"), "w") as f:
        json.dump({r["uid"]: int(k) for r, k in zip(traindev, folds)}, f)
    print(f"rows: train+dev={n_td} test={len(test)} | fold sizes={np.bincount(folds).tolist()}", flush=True)

    # phase 1: frozen NLI scores, sharded over all devices, cached
    ce_path = os.path.join(cfg.out, "ce_scores.npy")
    if cfg.fresh or not os.path.exists(ce_path):
        run_jobs([("ce", i, len(devices)) for i in range(len(devices))], devices, cfg)
        lookup = {}
        for i in range(len(devices)):
            with open(os.path.join(cfg.out, f"ce_shard{i}.json")) as f:
                lookup.update({(a, b): s for a, b, s in json.load(f)})
        np.save(ce_path, np.array([lookup[(a, b)] for a, b in zip(T["story"], T["gloss"])], dtype=np.float32))
    print(f"CE scores ready ({time.time() - t0:.0f}s)", flush=True)

    # phase 2: folds x seeds, spread over devices; finished folds are skipped
    def done(k, s):
        return os.path.exists(os.path.join(cfg.out, f"fold{k}_seed{s}.npz"))
    jobs = [("fold", k, s) for s in seeds for k in range(cfg.folds) if cfg.fresh or not done(k, s)]
    print(f"{len(jobs)} fold jobs on {len(devices)} device(s)", flush=True)
    run_jobs(jobs, devices, cfg)

    aggregate(cfg, traindev, test, T, n_td, seeds, t0, names)


if __name__ == "__main__":
    main()
