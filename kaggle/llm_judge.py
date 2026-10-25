#!/usr/bin/env python
"""E14: zero-shot LLM judge (roadmap item 6), scored as a run that ensemble.py can blend.

An instruction-tuned LLM rates "how plausible is this sense of the homonym in this story" on 1-5.
No generation: the score is the expected rating sum_k k * p(k) over the next-token probabilities
of the digits 1..5, averaged over a few prompt wordings. One prompt shows both senses of the
setup (the pair structure, roadmap item 3).

Nothing is trained, so no fold can see its own labels: the "OOF" predictions are simply the
scores on train+dev. folds.json is still written (same homonym folds as E12/E13) so that
ensemble.py can check alignment and choose blend weights on OOF only.

    python llm_judge.py --data DATA --test_gold GOLD            # Kaggle, 2x T4
    python llm_judge.py --data ../data --model Qwen/Qwen2.5-0.5B-Instruct --limit_stories 3   # smoke test
    python llm_judge.py --backend vllm --tp 2 --model Qwen/Qwen2.5-32B-Instruct-AWQ        # larger judge

Backends: 'hf' splits the layers over the GPUs (device_map=auto), so only one GPU works at a time;
'vllm' splits every layer over the GPUs (tensor parallel) and caches the shared prompt prefix.
"""
import argparse
import glob
import json
import os
import shutil
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
if not os.path.isdir(os.path.join(os.path.dirname(HERE), "ambistory")):
    # Kaggle script kernel: the package is uploaded flattened as ambistory__<module>.py
    flat = glob.glob("/kaggle/input/**/ambistory__data.py", recursive=True)
    if flat:
        pkg = "/tmp/code/ambistory"
        os.makedirs(pkg, exist_ok=True)
        for p in glob.glob(os.path.join(os.path.dirname(flat[0]), "ambistory__*.py")):
            shutil.copy(p, os.path.join(pkg, os.path.basename(p)[len("ambistory__"):]))
        sys.path.insert(0, "/tmp/code")
from ambistory import data as D  # noqa: E402
from ambistory import metrics as M  # noqa: E402
from ambistory import submission as S  # noqa: E402

SYSTEM = "You are a careful annotator of word meanings in short stories. Answer with a single digit."

SCALE = ("1 = the meaning does not fit at all, 2 = barely plausible, 3 = somewhat plausible, "
         "4 = plausible, 5 = clearly the intended meaning.")


def prompt_plain(r, other):
    return (f"Story:\n{D.story_text(r)}\n\n"
            f"In this story, how plausible is it that the word \"{r['homonym']}\" has this meaning:\n"
            f"\"{D.meaning_text(r)}\"\n\n{SCALE}\nRating (1-5):")


def prompt_pair(r, other):
    return (f"Story:\n{D.story_text(r)}\n\n"
            f"The word \"{r['homonym']}\" in the sentence \"{r['sentence'].strip()}\" is ambiguous. "
            f"It could mean:\n(A) {r['judged_meaning'].strip()}\n(B) {other['judged_meaning'].strip()}\n\n"
            f"Taking the whole story into account, how plausible is meaning (A)?\n{SCALE}\nRating (1-5):")


def prompt_readers(r, other):
    ending = "The story has no ending yet, so judge from the context so far. " if not r["ending"] else ""
    return (f"Five people read this story and rate how well a given meaning of \"{r['homonym']}\" fits it.\n\n"
            f"Story:\n{D.story_text(r)}\n\nMeaning: {D.meaning_text(r)}\n\n"
            f"{ending}What is their average rating, rounded to a whole number?\n{SCALE}\nRating (1-5):")


PROMPTS = {"plain": prompt_plain, "pair": prompt_pair, "readers": prompt_readers}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=None)
    p.add_argument("--out", default="/kaggle/working/e14" if os.path.isdir("/kaggle/working") else "outputs/e14")
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--prompts", default=",".join(PROMPTS))
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--backend", choices=["hf", "vllm"], default="hf")
    p.add_argument("--tp", type=int, default=0, help="vllm tensor-parallel size (0 = all GPUs)")
    p.add_argument("--max_len", type=int, default=1024, help="vllm context length")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--test_gold", default=None, help="labelled test.json, used ONLY for the final report")
    p.add_argument("--limit_stories", type=int, default=0, help="smoke test: keep only N stories per split")
    return p.parse_args(argv)


def find_data_dir(given):
    if given:
        return given
    for path in sorted(glob.glob("/kaggle/input/**/train.json", recursive=True)):
        d = os.path.dirname(path)
        if all(os.path.exists(os.path.join(d, f)) for f in ("dev.json", "test.json")):
            return d
    raise FileNotFoundError("no dir with train/dev/test.json under /kaggle/input; pass --data")


def _limit(rows, n):
    if not n:
        return rows
    keep = set(list(dict.fromkeys(D.story_id(r) for r in rows))[:n])
    return [r for r in rows if D.story_id(r) in keep]


def load_gold(path, test):
    if not path:
        return None
    gold = {r["id"]: r for r in D.load_split(path)}
    gold_rows = [gold[r["id"]] for r in test]
    assert all(g["judged_meaning"] == r["judged_meaning"] and g["sentence"] == r["sentence"]
               for g, r in zip(gold_rows, test)), "test_gold rows do not match test.json"
    return gold_rows if D.has_labels(gold_rows) else None


def other_sense(rows):
    """For each row, the row with the other sense in the same setup."""
    by_setup = {}
    for i, r in enumerate(rows):
        by_setup.setdefault(D.setup_id(r), []).append(i)
    out = []
    for i, r in enumerate(rows):
        mates = [j for j in by_setup[D.setup_id(r)] if j != i]
        out.append(rows[mates[0]] if mates else r)
    return out


@torch.no_grad()
def score(model, tok, texts, digit_ids, batch):
    """Probabilities of the digits 1..5 as the next token, shape (n, 5)."""
    order = np.argsort([-len(t) for t in texts])  # longest first: OOM shows up immediately
    probs = np.zeros((len(texts), 5), dtype=np.float32)
    dev = model.get_input_embeddings().weight.device
    for b in range(0, len(texts), batch):
        idx = order[b:b + batch]
        enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True).to(dev)
        logits = model(**enc).logits[:, -1, :].float()
        p = torch.softmax(logits[:, digit_ids], dim=-1).cpu().numpy()
        if not np.isfinite(p).all():
            raise FloatingPointError("non-finite digit probabilities (fp16 overflow?)")
        probs[idx] = p
        if b // batch % 25 == 0:
            print(f"  {b + len(idx)}/{len(texts)}", flush=True)
    return probs


def score_vllm(llm, texts, digit_strs):
    """Same as score() with vLLM. Only the top-20 next tokens come back, so digits outside them get
    probability ~0; the mass the digits hold is reported to check that this is harmless."""
    from vllm import SamplingParams
    params = SamplingParams(max_tokens=1, temperature=0.0, logprobs=20)
    probs = np.full((len(texts), 5), 1e-9, dtype=np.float64)
    mass = np.zeros(len(texts))
    for i, out in enumerate(llm.generate(texts, params)):
        top = out.outputs[0].logprobs[0]
        for lp in top.values():
            tok = (lp.decoded_token or "").strip()
            if tok in digit_strs:
                probs[i, digit_strs.index(tok)] += np.exp(lp.logprob)
        mass[i] = probs[i].sum()
    print(f"  digit mass in top-20: mean={mass.mean():.3f} min={mass.min():.3f}", flush=True)
    return (probs / probs.sum(1, keepdims=True)).astype(np.float32), float(mass.mean())


def main():
    cfg = parse_args()
    cfg.data = find_data_dir(cfg.data)
    if not cfg.test_gold:  # the labelled test.json lives in the inlp-project dataset on Kaggle
        gold = [p for p in glob.glob("/kaggle/input/**/test.json", recursive=True) if "inlp-project" in p]
        cfg.test_gold = gold[0] if gold else None
    os.makedirs(cfg.out, exist_ok=True)
    from transformers import AutoTokenizer
    import transformers
    t0 = time.time()

    load = lambda name: _limit(D.load_split(os.path.join(cfg.data, name)), cfg.limit_stories)  # noqa: E731
    traindev, test = load("train.json") + load("dev.json"), load("test.json")
    rows, n_td = traindev + test, len(traindev)
    folds = D.homonym_folds(traindev, cfg.folds)
    with open(os.path.join(cfg.out, "folds.json"), "w") as f:
        json.dump({r["uid"]: int(k) for r, k in zip(traindev, folds)}, f)

    gpus = torch.cuda.device_count()
    print(f"data={cfg.data} out={cfg.out} model={cfg.model} gpus={gpus} "
          f"torch={torch.__version__} transformers={transformers.__version__}", flush=True)
    tok = AutoTokenizer.from_pretrained(cfg.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if cfg.backend == "vllm":
        from vllm import LLM
        model = LLM(cfg.model, tensor_parallel_size=cfg.tp or gpus, dtype="half", max_model_len=cfg.max_len,
                    gpu_memory_utilization=0.9, enable_prefix_caching=True)
    else:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model, torch_dtype=torch.float16 if gpus else torch.float32,
            device_map="auto" if gpus else None).eval()
    digit_ids = [tok.encode(str(k), add_special_tokens=False) for k in range(1, 6)]
    assert all(len(d) == 1 for d in digit_ids), f"digits are not single tokens: {digit_ids}"
    digit_ids = [d[0] for d in digit_ids]
    print(f"model loaded ({time.time() - t0:.0f}s), digit ids {digit_ids}", flush=True)

    others = other_sense(rows)
    names = cfg.prompts.split(",")
    probs, digit_mass = {}, {}
    for name in names:
        texts = [tok.apply_chat_template([{"role": "system", "content": SYSTEM},
                                          {"role": "user", "content": PROMPTS[name](r, o)}],
                                         tokenize=False, add_generation_prompt=True)
                 for r, o in zip(rows, others)]
        if name == names[0]:
            print(f"example prompt:\n{texts[0]}", flush=True)
        print(f"prompt '{name}': {len(texts)} rows", flush=True)
        if cfg.backend == "vllm":
            probs[name], digit_mass[name] = score_vllm(model, texts, [str(k) for k in range(1, 6)])
        else:
            probs[name] = score(model, tok, texts, digit_ids, cfg.batch)
    np.savez(os.path.join(cfg.out, "probs.npz"), **probs)

    levels = np.arange(1, 6, dtype=np.float32)
    per_prompt = {n: probs[n] @ levels for n in names}
    pred = np.mean([per_prompt[n] for n in names], 0)
    oof, test_pred = pred[:n_td], pred[n_td:]
    np.save(os.path.join(cfg.out, "oof.npy"), oof)
    np.save(os.path.join(cfg.out, "test_pred.npy"), test_pred)
    S.write_predictions([r["id"] for r in test], test_pred, os.path.join(cfg.out, "predictions.jsonl"))

    report = {"model": cfg.model, "backend": cfg.backend, "prompts": names, "digit_mass_top20": digit_mass,
              "minutes": (time.time() - t0) / 60,
              "oof": M.evaluate(oof, traindev),
              "oof_per_prompt": {n: M.evaluate(per_prompt[n][:n_td], traindev) for n in names},
              "per_fold": [M.evaluate(oof[np.array(folds) == k], [r for r, f in zip(traindev, folds) if f == k])
                           for k in range(cfg.folds)]}
    gold = load_gold(cfg.test_gold, test)
    if gold:
        report["test"] = M.evaluate(test_pred, gold)
        report["test_per_prompt"] = {n: M.evaluate(per_prompt[n][n_td:], gold) for n in names}
    with open(os.path.join(cfg.out, "metrics.json"), "w") as f:
        json.dump(report, f, indent=2)
    for n in names:
        m = report["oof_per_prompt"][n]
        print(f"OOF  {n:>8}: rho={m['spearman']:.4f} acc={m['accuracy']:.4f}")
    print(f"OOF  {'mean':>8}: rho={report['oof']['spearman']:.4f} acc={report['oof']['accuracy']:.4f}")
    if gold:
        print(f"TEST {'mean':>8}: rho={report['test']['spearman']:.4f} acc={report['test']['accuracy']:.4f}")
    print(f"done in {report['minutes']:.1f} min")


if __name__ == "__main__":
    main()
