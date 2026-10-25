#!/usr/bin/env python
"""Ablations of the ideas tried in E01-E12, re-tested under the honest protocol.

Each ablation changes ONE thing relative to a reference run (fixed E12 or E13, seed 42, same
homonym-grouped folds) and runs as its own Kaggle kernel. Run locally:

    python ablations.py generate --dir kernels/           # one kernel folder per ablation
    python ablations.py run      --dir kernels/ --max 2   # push, keep <= 2 running, wait for all
    python ablations.py collect  --dir kernels/ --dest ../results/ablations
"""
import argparse
import json
import os
import subprocess
import sys
import time

OWNER = "shubhadeepmandal"
DATASETS = [f"{OWNER}/ambistory-e12", f"{OWNER}/inlp-project"]
E12_REF = "results/e12_fixed_kaggle_v1"            # fixed E12, seed 42
E13_REF = "results/e13_deberta_large_6ep_kaggle_v3"  # E13 large, 6 epochs, seed 42
E13_BASE = ["--epochs", "6"]

# name, script, extra args, reference (results dir or another ablation name), question answered
ABLATIONS = [
    ("a12-hom-mean", "e12", ["--hom_mean"], E12_REF,
     "C3: does the per-homonym mean feature (E12) help?"),
    ("a12-nons-meta", "e12", ["--nons_meta"], E12_REF,
     "C1: what did the 'nonsensical' meta feature (E07-E09) do on test?"),
    ("a12-select-on-outer", "e12", ["--select_on_outer"], E12_REF,
     "C4: how optimistic is picking the epoch on the scored fold?"),
    ("a12-random-batches", "e12", ["--random_batches"], E12_REF,
     "H1: does the ranking loss matter once batches are random?"),
    ("a12-no-swa", "e12", ["--swa_start", "2.0"], E12_REF, "Is SWA (E12) useful?"),
    ("a12-no-nli-score", "e12", ["--no_ce"], E12_REF, "Is the frozen NLI entailment score useful?"),
    ("a12-frozen-encoder", "e12", ["--ft_epochs", "0"], E12_REF, "Is fine-tuning MPNet per fold useful?"),
    ("a13-regression-mse", "e13", E13_BASE + ["--head", "regression", "--kl_w", "0", "--mse_w", "1", "--rank_w", "0"],
     E13_REF, "Stage 1 objective: plain regression with MSE"),
    ("a13-regression-huber", "e13", E13_BASE + ["--head", "regression", "--kl_w", "0", "--mse_w", "1", "--rank_w", "0",
                                                 "--reg_loss", "huber"], "a13-regression-mse", "E05: Huber vs MSE"),
    ("a13-regression-sdweight", "e13", E13_BASE + ["--head", "regression", "--kl_w", "0", "--mse_w", "1", "--rank_w",
                                                    "0", "--sd_weight"], "a13-regression-mse", "E04: 1/sd^2-weighted MSE vs MSE"),
    ("a13-regression-e06rank", "e13", E13_BASE + ["--head", "regression", "--kl_w", "0", "--mse_w", "0.5", "--rank_w",
                                                   "0.5", "--rank_type", "e06"], "a13-regression-mse",
     "E06: MSE + Model 6's global ranking loss vs MSE"),
    ("a13-no-rank", "e13", E13_BASE + ["--rank_w", "0"], E13_REF, "E07: is the within-setup ranking loss useful?"),
    ("a13-random-batches", "e13", E13_BASE + ["--batching", "random"], E13_REF,
     "H1: ranking loss with random batches (as in E10-E12)"),
    ("a13-no-kl", "e13", E13_BASE + ["--kl_w", "0", "--mse_w", "1"], E13_REF,
     "E08: is the annotator-distribution KL target useful?"),
    ("a13-no-example", "e13", E13_BASE + ["--no_example"], E13_REF, "Is the example sentence in the input useful?"),
    ("a13-no-mark", "e13", E13_BASE + ["--no_mark"], E13_REF, "Is quoting the homonym useful?"),
    ("a13-base-size", "e13", E13_BASE + ["--model", "microsoft/deberta-v3-base"], E13_REF, "large vs base backbone"),
    ("a13-nli-base", "e13", E13_BASE + ["--model", "cross-encoder/nli-deberta-v3-base"], "a13-base-size",
     "E10/E11 claim: does NLI initialisation help (same size)?"),
]

RUNNER = '''"""Ablation {name}: {question}"""
import glob, os, shutil, subprocess, sys

src = os.path.dirname(glob.glob("/kaggle/input/**/e13_train.py", recursive=True)[0])
code = "/tmp/code"
for d in ("ambistory", "kaggle", "data"):
    os.makedirs(os.path.join(code, d), exist_ok=True)
for path in glob.glob(os.path.join(src, "ambistory__*.py")):
    shutil.copy(path, os.path.join(code, "ambistory", os.path.basename(path)[len("ambistory__"):]))
for f in ("e12_train.py", "e13_train.py", "ensemble.py"):
    shutil.copy(os.path.join(src, f), os.path.join(code, "kaggle", f))
for f in ("train.json", "dev.json", "test.json"):
    shutil.copy(os.path.join(src, f), os.path.join(code, "data", f))
gold = [p for p in glob.glob("/kaggle/input/**/test.json", recursive=True) if "inlp-project" in p]
cmd = [sys.executable, os.path.join(code, "kaggle", "{script}_train.py"), "--data", os.path.join(code, "data"),
       "--out", "/kaggle/working/{name}"] + {args!r} + (["--test_gold", gold[0]] if gold else [])
print(" ".join(cmd), flush=True)
subprocess.run(cmd, check=True)
'''


def slug(name):
    return f"ambistory-abl-{name}"


def kernel_id(name):
    return f"{OWNER}/{slug(name)}"


def generate(cfg):
    for name, script, args, _, question in ABLATIONS:
        d = os.path.join(cfg.dir, name)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "run.py"), "w") as f:
            f.write(RUNNER.format(name=name, question=question, script=script, args=args))
        meta = {"id": kernel_id(name), "title": slug(name).replace("-", " "), "code_file": "run.py",
                "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": True,
                "enable_internet": True, "dataset_sources": DATASETS, "competition_sources": [], "kernel_sources": []}
        with open(os.path.join(d, "kernel-metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)
    print(f"generated {len(ABLATIONS)} kernels in {cfg.dir}")


def kaggle(cfg, *args):
    r = subprocess.run([cfg.kaggle, *args], capture_output=True, text=True)
    return (r.stdout + r.stderr).strip()


def status(cfg, kid):
    out = kaggle(cfg, "kernels", "status", kid)
    for s in ("COMPLETE", "ERROR", "CANCEL", "RUNNING", "QUEUED"):
        if s in out.upper():
            return s
    return "UNKNOWN"


def run(cfg):
    names = [a[0] for a in ABLATIONS if not cfg.only or a[0] in cfg.only.split(",")]
    state = {n: "PENDING" for n in names}
    while True:
        busy = sum(status(cfg, k) in ("RUNNING", "QUEUED") for k in cfg.watch.split(",") if k)
        for n in names:
            if state[n] in ("RUNNING", "QUEUED", "PUSHED"):
                state[n] = status(cfg, kernel_id(n))
                if state[n] == "UNKNOWN":
                    state[n] = "PUSHED"
        busy += sum(s in ("RUNNING", "QUEUED", "PUSHED") for s in state.values())
        for n in names:
            if state[n] == "PENDING" and busy < cfg.max:
                out = kaggle(cfg, "kernels", "push", "-p", os.path.join(cfg.dir, n), "--accelerator", "NvidiaTeslaT4")
                if "successfully pushed" in out:
                    state[n], busy = "PUSHED", busy + 1
                    print(f"{time.strftime('%H:%M:%S')} pushed {n}", flush=True)
                else:
                    print(f"{time.strftime('%H:%M:%S')} push of {n} refused, will retry: {out[-200:]}", flush=True)
                    break
        line = " ".join(f"{n}={s}" for n, s in state.items() if s not in ("PENDING",))
        print(f"{time.strftime('%H:%M:%S')} {line}", flush=True)
        if all(s in ("COMPLETE", "ERROR", "CANCEL") for s in state.values()):
            break
        time.sleep(cfg.poll)
    print("FINAL " + json.dumps(state))


def load_metrics(path):
    with open(os.path.join(path, "metrics.json")) as f:
        m = json.load(f)
    test = m.get("test_without_rank_blend", {})
    return {"oof_rho": m["oof"]["spearman"], "oof_acc": m["oof"]["accuracy"],
            "test_rho": test.get("spearman"), "test_acc": test.get("accuracy")}


def collect(cfg):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results, rows = {}, []
    for name, script, args, ref, question in ABLATIONS:
        dest = os.path.join(cfg.dest, name)
        if not os.path.exists(os.path.join(dest, name, "metrics.json")):
            os.makedirs(dest, exist_ok=True)
            print(kaggle(cfg, "kernels", "output", kernel_id(name), "-p", dest).splitlines()[-1:])
        try:
            results[name] = load_metrics(os.path.join(dest, name))
        except (FileNotFoundError, KeyError) as e:
            print(f"{name}: no metrics ({e})")
    for name, script, args, ref, question in ABLATIONS:
        if name not in results:
            rows.append({"ablation": name, "question": question, "status": "missing"})
            continue
        r = results[name]
        base = results.get(ref) if ref in results else load_metrics(os.path.join(root, ref))
        row = {"ablation": name, "question": question, "reference": ref, **{k: round(v, 4) for k, v in r.items() if v is not None}}
        for k in ("oof_rho", "oof_acc", "test_rho", "test_acc"):
            if r.get(k) is not None and base.get(k) is not None:
                row["d_" + k] = round(r[k] - base[k], 4)
        rows.append(row)
    with open(os.path.join(cfg.dest, "ablations.json"), "w") as f:
        json.dump(rows, f, indent=2)
    print(f"{'ablation':26} {'OOF rho':>8} {'dOOF':>7} {'test rho':>8} {'dtest':>7}  question")
    for r in rows:
        if "oof_rho" in r:
            print(f"{r['ablation']:26} {r['oof_rho']:8.4f} {r.get('d_oof_rho', 0):+7.4f} "
                  f"{r.get('test_rho', float('nan')):8.4f} {r.get('d_test_rho', float('nan')):+7.4f}  {r['question']}")
        else:
            print(f"{r['ablation']:26} missing")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["generate", "run", "collect"])
    ap.add_argument("--dir", default="kernels")
    ap.add_argument("--dest", default=os.path.join("..", "results", "ablations"))
    ap.add_argument("--kaggle", default="kaggle", help="path to the kaggle CLI")
    ap.add_argument("--max", type=int, default=2, help="max kernels running at once (incl. --watch)")
    ap.add_argument("--watch", default="", help="other kernel ids that count toward --max")
    ap.add_argument("--only", default="", help="comma-separated subset of ablation names")
    ap.add_argument("--poll", type=int, default=120)
    cfg = ap.parse_args()
    {"generate": generate, "run": run, "collect": collect}[cfg.cmd](cfg)


if __name__ == "__main__":
    sys.exit(main())
