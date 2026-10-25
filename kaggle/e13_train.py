#!/usr/bin/env python
"""E13: DeBERTa-v3-large cross-encoder for AmbiStory, one fold per GPU (Kaggle T4 x2).

Retries the cross-encoder of E10 with the causes of its failure removed:
  * large instead of base backbone, mean pooling, ordinal (proportional-odds) head;
  * input pair = story with the homonym quoted  /  "homonym: gloss. Example: ..."
    (E10 never saw the example sentence or knew which word was meant);
  * story-grouped batches so the within-setup ranking loss actually fires (H1);
  * KL to the annotator distribution + ranking + MSE on the expected score;
  * layer-wise LR decay, fp16 autocast, gradient accumulation;
  * checkpoint chosen on an inner homonym-grouped split, never the outer fold (C4);
  * no stacked MLP, so the H5 issue of E12 does not arise.

Same folds, outputs and resume behaviour as e12_train.py:
    !python e13_train.py --data ... --test_gold ...

--deploy trains one model per seed on ALL of train+dev (checkpoint still chosen on an inner
homonym-grouped split) and saves it as fp16 safetensors + tokenizer + config for app.py.
"""
import argparse
import glob
import math
import os
import re
import shutil
import sys
import tempfile
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from ambistory import metrics as M  # noqa: E402
from ambistory.batching import RandomBatchSampler, StoryBatchSampler  # noqa: E402
from ambistory.losses import within_setup_ranking_loss  # noqa: E402
from e12_train import (OrdinalHead, aggregate, cosine_warmup, find_data_dir, get_devices,  # noqa: E402
                       inner_split, load_gold, prepare, run_jobs, soft_rank_blend)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=None)
    p.add_argument("--out", default="/kaggle/working/e13" if os.path.isdir("/kaggle/working") else "outputs/e13")
    p.add_argument("--model", default="microsoft/deberta-v3-large")
    p.add_argument("--model_cache", default=os.path.join(tempfile.gettempdir(), "e13_models"))
    p.add_argument("--gpus", default="auto")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seeds", default="42")
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--lr", type=float, default=1.5e-5, help="LR of the top encoder layer")
    p.add_argument("--llrd", type=float, default=0.9, help="per-layer LR decay toward the embeddings")
    p.add_argument("--head_lr", type=float, default=5e-4)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--warmup", type=float, default=0.1)
    p.add_argument("--stories_per_batch", type=int, default=2, help="x6 rows per micro-batch")
    p.add_argument("--accum", type=int, default=2, help="gradient accumulation steps")
    p.add_argument("--max_len", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--rank_w", type=float, default=0.5)
    p.add_argument("--rank_margin", type=float, default=0.3)
    p.add_argument("--mse_w", type=float, default=0.2, help="weight of the regression term (MSE or Huber)")
    # ablation switches (defaults = the E13 system)
    p.add_argument("--head", choices=["ordinal", "regression"], default="ordinal")
    p.add_argument("--kl_w", type=float, default=1.0, help="annotator-distribution KL (ordinal head only)")
    p.add_argument("--reg_loss", choices=["mse", "huber"], default="mse")
    p.add_argument("--sd_weight", action="store_true", help="weight the regression term by 1/max(sd,1)^2 (E04)")
    p.add_argument("--rank_type", choices=["setup", "e06"], default="setup",
                   help="'e06' = Model 6's global margin - dy*dp loss over all batch pairs")
    p.add_argument("--batching", choices=["story", "random"], default="story")
    p.add_argument("--no_example", action="store_true", help="drop the example sentence from the input")
    p.add_argument("--no_mark", action="store_true", help="do not quote the homonym in the story")
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--inner_val_frac", type=float, default=0.15)
    p.add_argument("--eval_batch", type=int, default=64)
    p.add_argument("--grad_ckpt", action="store_true", help="gradient checkpointing (less memory, ~30% slower)")
    p.add_argument("--rank_blend_w", type=float, default=0.15)
    p.add_argument("--test_gold", default=None, help="labelled test.json, used ONLY for the final report")
    p.add_argument("--limit_stories", type=int, default=0, help="smoke test: keep only N stories per split")
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--deploy", action="store_true", help="fit on all of train+dev and save the model for app.py")
    return p.parse_args(argv)


def resolve_backbone(name, cache_dir):
    """Local dir with safetensors weights. Repos that only ship pytorch_model.bin (deberta-v3-large)
    are converted once here, so workers never torch.load a pickle (newer transformers refuse that
    on torch < 2.6) and the unused 700 MB generator checkpoint is never downloaded."""
    if os.path.isdir(name):
        src = name
    else:
        from huggingface_hub import list_repo_files, snapshot_download
        files = list_repo_files(name)
        weights = "model.safetensors" if "model.safetensors" in files else "pytorch_model.bin"
        src = snapshot_download(name, allow_patterns=["*.json", "*.model", "*.txt", weights])
    if glob.glob(os.path.join(src, "*.safetensors")):
        return src
    dst = os.path.join(cache_dir, os.path.basename(os.path.normpath(name)))
    if not os.path.exists(os.path.join(dst, "model.safetensors")):
        from safetensors.torch import save_file
        os.makedirs(dst, exist_ok=True)
        for f in os.listdir(src):
            if not f.endswith(".bin"):
                shutil.copy(os.path.join(src, f), dst)
        state = torch.load(os.path.join(src, "pytorch_model.bin"), map_location="cpu", weights_only=True)
        save_file({k: v.contiguous().clone() for k, v in state.items()},
                  os.path.join(dst, "model.safetensors"), metadata={"format": "pt"})
    return dst


def build_inputs(rows, mark=True, example=True):
    """(story with the homonym quoted, 'homonym: gloss. Example: sentence') for every row."""
    a, b = [], []
    for r in rows:
        sentence = r["sentence"].strip()
        if mark:
            pattern = r"\b" + re.escape(r["homonym"]) + r"\b"
            sentence = re.sub(pattern, lambda m: f'"{m.group(0)}"', sentence, count=1, flags=re.I)
        a.append(" ".join(x for x in (r["precontext"].strip(), sentence, r["ending"]) if x))
        gloss = r["judged_meaning"].strip().rstrip(".")
        example = (r.get("example_sentence") or "").strip() if example else ""
        b.append(f'{r["homonym"]}: {gloss}.' + (f" Example: {example}" if example else ""))
    return a, b


class RegressionHead(nn.Module):
    """Plain scalar head (the stage 1-2 setup), initialised at the label mean. Returns (None, score)."""

    def __init__(self, in_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, 1)
        nn.init.zeros_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 3.0)

    def forward(self, h):
        return None, self.linear(h).squeeze(-1)


class CrossEncoder(nn.Module):
    def __init__(self, path, dropout, grad_ckpt, head="ordinal", pretrained=True):
        super().__init__()
        from transformers import AutoConfig, AutoModel
        # fp32 master weights: transformers>=5 keeps the checkpoint dtype (fp16 for deberta-v3-large),
        # and GradScaler cannot unscale fp16 gradients. autocast still runs the matmuls in fp16.
        if pretrained:
            self.backbone = AutoModel.from_pretrained(path).float()
        else:  # architecture only; weights come from a deployed state dict (load_deployed)
            self.backbone = AutoModel.from_config(AutoConfig.from_pretrained(path)).float()
        if grad_ckpt:
            self.backbone.gradient_checkpointing_enable()
        self.drop = nn.Dropout(dropout)
        hidden = self.backbone.config.hidden_size
        self.head = OrdinalHead(hidden) if head == "ordinal" else RegressionHead(hidden)

    def forward(self, enc):
        hidden = self.backbone(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = ((hidden * mask).sum(1) / mask.sum(1).clamp(min=1)).float()
        with torch.autocast(device_type="cuda", enabled=False):  # ordinal head in fp32
            return self.head(self.drop(pooled))


def param_groups(model, lr, decay, head_lr, wd):
    """Layer-wise LR decay: top layer gets lr, each layer below gets x decay; no decay on biases/norms."""
    n_layers = model.backbone.config.num_hidden_layers
    groups = {}
    for name, p in model.named_parameters():
        if name.startswith("head."):
            group_lr = head_lr
        else:
            m = re.search(r"encoder\.layer\.(\d+)\.", name)
            depth = int(m.group(1)) + 1 if m else (0 if "embeddings." in name else n_layers + 1)
            group_lr = lr * decay ** (n_layers + 1 - depth)
        groups.setdefault((group_lr, p.ndim == 1), []).append(p)
    return [{"params": ps, "lr": g_lr, "weight_decay": 0.0 if no_wd else wd}
            for (g_lr, no_wd), ps in groups.items()]


def tokenize(tok, A, B, idx, max_len, dev):
    return tok([A[i] for i in idx], [B[i] for i in idx], padding=True, truncation="longest_first",
               max_length=max_len, return_tensors="pt").to(dev)


@torch.no_grad()
def predict(model, tok, A, B, idx, cfg, dev):
    """Expected score for rows idx; length-sorted batches to minimise padding."""
    model.eval()
    order = sorted(range(len(idx)), key=lambda j: len(A[idx[j]]) + len(B[idx[j]]))
    out = np.empty(len(idx), dtype=np.float32)
    for s in range(0, len(order), cfg.eval_batch):
        js = order[s:s + cfg.eval_batch]
        enc = tokenize(tok, A, B, [idx[j] for j in js], cfg.max_len, dev)
        with torch.autocast("cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
            out[js] = model(enc)[1].cpu().numpy()
    return out


def training_loss(cfg, probs, exp, y, dist, gid, sd):
    """KL (ordinal head) + regression term + ranking term. Returns (loss, ranking pairs used)."""
    loss = exp.new_tensor(0.0)
    if probs is not None and cfg.kl_w > 0:
        loss = loss + cfg.kl_w * F.kl_div(torch.log(probs), dist, reduction="batchmean")
    if cfg.mse_w > 0:
        per_row = F.huber_loss(exp, y, reduction="none", delta=1.0) if cfg.reg_loss == "huber" \
            else (exp - y) ** 2
        if cfg.sd_weight:  # E04: 1/sd^2 with sd clamped at 1, normalised to mean 1 in the batch
            w = 1.0 / sd.clamp(min=1.0) ** 2
            per_row = per_row * (w / w.mean())
        loss = loss + cfg.mse_w * per_row.mean()
    n = 0
    if cfg.rank_w > 0:
        if cfg.rank_type == "setup":
            rank, n = within_setup_ranking_loss(exp, y, gid, margin=cfg.rank_margin, min_gap=0.1)
        else:  # Model 6: margin 1.0, scaled by the label gap, over every pair in the batch
            dy, dp = y[:, None] - y[None, :], exp[:, None] - exp[None, :]
            mask = dy.abs() > 0
            n = int(mask.sum())
            rank = (torch.clamp(1.0 - dy * dp, min=0) * mask).sum() / mask.sum().clamp(min=1)
        loss = loss + cfg.rank_w * rank
    return loss, n


def job_fold(fold, seed, ctx):
    from transformers import AutoTokenizer
    cfg, T, dev, log = ctx["cfg"], ctx["T"], ctx["dev"], ctx["log"]
    A, B, folds, n_td = ctx["A"], ctx["B"], ctx["folds"], ctx["n_td"]
    t_start = time.time()
    torch.manual_seed(seed + fold)
    np.random.seed(seed + fold)
    outer_tr, outer_va = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)  # fold -1: deploy, all rows
    test_idx = np.arange(n_td, len(A))
    in_tr, in_va = inner_split(ctx["traindev"], outer_tr, cfg.inner_val_frac, seed * 100 + fold)
    log(f"fold {fold} seed {seed}: inner-train={len(in_tr)} inner-val={len(in_va)} outer-val={len(outer_va)}")

    tok = AutoTokenizer.from_pretrained(cfg.model_path)
    model = CrossEncoder(cfg.model_path, cfg.dropout, cfg.grad_ckpt, cfg.head).to(dev)
    opt = torch.optim.AdamW(param_groups(model, cfg.lr, cfg.llrd, cfg.head_lr, cfg.wd), eps=1e-6)
    if cfg.batching == "story":
        sampler = StoryBatchSampler(T["story_gid"][in_tr], cfg.stories_per_batch, seed=seed + fold)
    else:  # same number of rows per batch, but shuffled rows (ranking pairs rarely share a batch)
        sampler = RandomBatchSampler(len(in_tr), 6 * cfg.stories_per_batch, seed=seed + fold)
    total = cfg.epochs * math.ceil(len(sampler) / cfg.accum)
    sched = cosine_warmup(opt, int(cfg.warmup * total), total)
    amp = dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    y = torch.tensor(T["y"], device=dev)
    dist = torch.tensor(0.9 * T["dist"] + 0.02, device=dev)  # label smoothing toward uniform
    gid = torch.tensor(T["setup"][:n_td], device=dev)
    sd = torch.tensor(T["sd"], device=dev)

    best, best_state, bad = -2.0, None, 0
    for ep in range(1, cfg.epochs + 1):
        model.train()
        t0, n_pairs, loss_sum, skipped = time.time(), 0, 0.0, 0
        batches = list(sampler)
        opt.zero_grad(set_to_none=True)
        for bi, b in enumerate(batches):
            idx = in_tr[b]
            it = torch.tensor(idx, device=dev)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                probs, exp = model(tokenize(tok, A, B, idx, cfg.max_len, dev))
            loss, n = training_loss(cfg, probs, exp, y[it], dist[it], gid[it], sd[it])
            if not torch.isfinite(loss):
                skipped += 1
                continue
            scaler.scale(loss / cfg.accum).backward()
            if (bi + 1) % cfg.accum == 0 or bi == len(batches) - 1:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
            n_pairs += n
            loss_sum += loss.item()
        p_va = predict(model, tok, A, B, in_va, cfg, dev)
        sp = M.spearman(p_va, T["y"][in_va])
        log(f"  ep {ep}/{cfg.epochs} loss={loss_sum / max(len(batches) - skipped, 1):.4f} "
            f"inner-val rho={sp:.4f} rank-pairs={n_pairs} skipped={skipped} ({(time.time() - t0) / 60:.1f} min)")
        if sp > best:
            best, bad = sp, 0
            best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.patience:
                log(f"  early stop at epoch {ep}")
                break
    model.load_state_dict(best_state)

    if fold < 0:
        save_deployed(model, tok, cfg, seed, best, np.clip(predict(model, tok, A, B, test_idx, cfg, dev), 1.0, 5.0))
        log(f"deploy seed {seed}: saved ({(time.time() - t_start) / 60:.1f} min)")
        return
    pred_va = np.clip(predict(model, tok, A, B, outer_va, cfg, dev), 1.0, 5.0)
    pred_te = np.clip(predict(model, tok, A, B, test_idx, cfg, dev), 1.0, 5.0)
    m = M.evaluate(pred_va, [ctx["traindev"][i] for i in outer_va])
    log(f"fold {fold} seed {seed}: OUTER rho={m['spearman']:.4f} acc={m['accuracy']:.4f} "
        f"({(time.time() - t_start) / 60:.1f} min)")
    name = os.path.join(cfg.out, f"fold{fold}_seed{seed}")
    np.savez(name + ".tmp.npz", val_idx=outer_va, val_pred=pred_va, test_pred=pred_te, inner_best=best)
    os.replace(name + ".tmp.npz", name + ".npz")
    del model, opt, best_state
    if amp:
        torch.cuda.empty_cache()


def save_deployed(model, tok, cfg, seed, inner_best, test_pred):
    """fp16 weights + tokenizer + backbone config, enough to rebuild CrossEncoder offline (app.py)."""
    import json
    from safetensors.torch import save_file
    d = os.path.join(cfg.out, f"model_seed{seed}")
    os.makedirs(d, exist_ok=True)
    state = {k: v.detach().to("cpu", torch.float16).contiguous() for k, v in model.state_dict().items()}
    save_file(state, os.path.join(d, "model.safetensors"), metadata={"format": "pt"})
    model.backbone.config.save_pretrained(d)
    tok.save_pretrained(d)
    with open(os.path.join(d, "e13_config.json"), "w") as f:
        json.dump({"head": cfg.head, "max_len": cfg.max_len, "mark": not cfg.no_mark,
                   "example": not cfg.no_example, "rank_blend_w": cfg.rank_blend_w,
                   "seed": seed, "epochs": cfg.epochs, "inner_val_rho": float(inner_best)}, f, indent=2)
    np.save(os.path.join(d, "test_pred.npy"), test_pred)


def load_deployed(d, device="cpu"):
    """Rebuild a model saved by save_deployed: returns (model, tokenizer, e13_config). No download."""
    import json
    from safetensors.torch import load_file
    from transformers import AutoTokenizer
    with open(os.path.join(d, "e13_config.json")) as f:
        meta = json.load(f)
    model = CrossEncoder(d, 0.0, False, meta["head"], pretrained=False)
    model.load_state_dict({k: v.float() for k, v in load_file(os.path.join(d, "model.safetensors")).items()})
    return model.to(device).eval(), AutoTokenizer.from_pretrained(d), meta


def worker(gpu, queue, cfg):
    dev = torch.device(f"cuda:{gpu}" if gpu is not None else "cpu")
    if dev.type == "cuda":
        torch.cuda.set_device(dev)
    tag = f"[{'gpu' + str(gpu) if gpu is not None else 'cpu'}]"
    log = lambda msg: print(f"{time.strftime('%H:%M:%S')} {tag} {msg}", flush=True)  # noqa: E731
    traindev, test, T, n_td, folds = prepare(cfg)
    A, B = build_inputs(traindev + test, mark=not cfg.no_mark, example=not cfg.no_example)
    ctx = {"cfg": cfg, "T": T, "dev": dev, "log": log, "folds": folds, "n_td": n_td,
           "traindev": traindev, "A": A, "B": B}
    while True:
        job = queue.get()
        if job is None:
            break
        job_fold(job[1], job[2], ctx)


def main():
    import json
    cfg = parse_args()
    cfg.data_dir = find_data_dir(cfg.data)
    os.makedirs(cfg.out, exist_ok=True)
    seeds = [int(s) for s in cfg.seeds.split(",")]
    devices, names = get_devices(cfg.gpus)
    import transformers
    print(f"data={cfg.data_dir}\nout={cfg.out}\ndevices={names}\n"
          f"torch={torch.__version__} transformers={transformers.__version__}", flush=True)
    t0 = time.time()
    cfg.model_path = resolve_backbone(cfg.model, cfg.model_cache)
    print(f"backbone ready at {cfg.model_path} ({time.time() - t0:.0f}s)", flush=True)

    traindev, test, T, n_td, folds = prepare(cfg)
    with open(os.path.join(cfg.out, "folds.json"), "w") as f:
        json.dump({r["uid"]: int(k) for r, k in zip(traindev, folds)}, f)
    A, _ = build_inputs(traindev[:1])
    print(f"rows: train+dev={n_td} test={len(test)} | example input: {A[0][:160]}...", flush=True)

    def done(k, s):
        return os.path.exists(os.path.join(cfg.out, f"fold{k}_seed{s}.npz"))
    if cfg.deploy:
        run_jobs([("deploy", -1, s) for s in seeds], devices, cfg, target=worker)
        report_deployed(cfg, test, seeds)
        return
    jobs = [("fold", k, s) for s in seeds for k in range(cfg.folds) if cfg.fresh or not done(k, s)]
    print(f"{len(jobs)} fold jobs on {len(devices)} device(s)", flush=True)
    run_jobs(jobs, devices, cfg, target=worker)
    aggregate(cfg, traindev, test, T, n_td, seeds, t0, names)


def report_deployed(cfg, test, seeds):
    import json
    preds = np.stack([np.load(os.path.join(cfg.out, f"model_seed{s}", "test_pred.npy")) for s in seeds])
    gold = load_gold(cfg.test_gold, test)
    report = {"seeds": seeds}
    if gold:
        report["test_per_seed"] = [M.evaluate(p, gold) for p in preds]
        report["test_mean"] = M.evaluate(preds.mean(0), gold)
        gid = np.array(group_ids(test))
        report["test_mean_rank_blend"] = M.evaluate(soft_rank_blend(preds.mean(0), gid, cfg.rank_blend_w), gold)
    with open(os.path.join(cfg.out, "deploy_metrics.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2), flush=True)


def group_ids(rows):
    from ambistory import data as D
    ids = {}
    return [ids.setdefault(D.setup_id(r), len(ids)) for r in rows]


if __name__ == "__main__":
    main()
