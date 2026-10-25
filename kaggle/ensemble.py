#!/usr/bin/env python
"""Ensemble finished runs (e.g. E12 + E13) and calibrate for accuracy, choosing everything on OOF.

    python ensemble.py --runs /kaggle/input/.../e12 /kaggle/working/e13 --data DATA --test_gold GOLD

Each run dir needs folds.json, oof.npy and fold*_seed*.npz (as written by e12/e13_train.py).
Steps, each picked only on OOF predictions:
  1. weighted average of the runs (weight grid for 2 runs, equal weights otherwise);
  2. soft-rank blend within setups, kept only if it raises OOF Spearman;
  3. monotone linear calibration p' = mean + s*(p - mean) + c chosen for OOF accuracy.
     A strictly increasing map leaves Spearman unchanged (apart from ties created by clipping to [1, 5]).
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from ambistory import data as D  # noqa: E402
from ambistory import metrics as M  # noqa: E402
from ambistory import submission as S  # noqa: E402
from e12_train import _limit, load_gold, soft_rank_blend  # noqa: E402


def load_run(path):
    with open(os.path.join(path, "folds.json")) as f:
        folds = json.load(f)
    oof = np.load(os.path.join(path, "oof.npy"))
    if os.path.exists(os.path.join(path, "test_pred.npy")):
        test = np.load(os.path.join(path, "test_pred.npy"))
    else:  # older runs: average the per-fold test predictions over folds and seeds
        test = np.mean([np.load(p)["test_pred"] for p in glob.glob(os.path.join(path, "fold*_seed*.npz"))], 0)
    return folds, oof, test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--test_gold", default=None)
    ap.add_argument("--out", default="ensemble_out")
    ap.add_argument("--rank_blend_w", type=float, default=0.15)
    ap.add_argument("--limit_stories", type=int, default=0, help="must match the runs (smoke tests only)")
    cfg = ap.parse_args()
    os.makedirs(cfg.out, exist_ok=True)

    load = lambda name: _limit(D.load_split(os.path.join(cfg.data, name)), cfg.limit_stories)  # noqa: E731
    traindev, test = load("train.json") + load("dev.json"), load("test.json")
    runs = [load_run(p) for p in cfg.runs]
    for p, (folds, oof, te) in zip(cfg.runs, runs):
        assert folds == runs[0][0], f"{p} used different folds than {cfg.runs[0]}"
        assert len(oof) == len(traindev) and len(te) == len(test), f"{p} does not match the data"
    ids = {s: i for i, s in enumerate(dict.fromkeys(D.setup_id(r) for r in traindev + test))}
    gid_oof = np.array([ids[D.setup_id(r)] for r in traindev])
    gid_test = np.array([ids[D.setup_id(r)] for r in test])
    oofs, tests = np.stack([r[1] for r in runs]), np.stack([r[2] for r in runs])
    report = {"runs": cfg.runs, "single_runs_oof": [M.evaluate(o, traindev) for o in oofs]}

    # 1. weights
    if len(runs) == 2:
        grid = [np.array([w, 1 - w]) for w in np.round(np.arange(0, 1.0001, 0.05), 2)]
    else:
        grid = [np.full(len(runs), 1 / len(runs))]
    scored = [(M.spearman(w @ oofs, [r["average"] for r in traindev]), w) for w in grid]
    best_sp, weights = max(scored, key=lambda t: t[0])
    oof, te = weights @ oofs, weights @ tests
    report["weights"] = weights.tolist()
    report["oof_weighted"] = M.evaluate(oof, traindev)

    # 2. rank blend
    oof_b = soft_rank_blend(oof, gid_oof, cfg.rank_blend_w)
    use_blend = M.evaluate(oof_b, traindev)["spearman"] > report["oof_weighted"]["spearman"]
    if use_blend:
        oof, te = oof_b, soft_rank_blend(te, gid_test, cfg.rank_blend_w)
    report["rank_blend_used"] = bool(use_blend)

    # 3. calibration for accuracy
    avg = np.array([r["average"] for r in traindev])
    std = np.array([r["stdev"] for r in traindev])
    mean = float(oof.mean())
    best = (-1.0, 1.0, 0.0)
    for s in np.arange(0.8, 3.01, 0.05):
        for c in np.arange(-0.75, 0.751, 0.05):
            acc = M.accuracy_within_sd(np.clip(mean + s * (oof - mean) + c, 1, 5), avg, std)
            if acc > best[0] + 1e-9:
                best = (acc, float(s), float(c))
    _, s, c = best
    oof_cal = np.clip(mean + s * (oof - mean) + c, 1, 5)
    te_cal = np.clip(mean + s * (te - mean) + c, 1, 5)
    report["calibration"] = {"scale": s, "shift": c, "center": mean}
    report["oof_final"] = M.evaluate(oof_cal, traindev)

    S.write_predictions([r["id"] for r in test], te_cal, os.path.join(cfg.out, "predictions.jsonl"))
    S.check_against(os.path.join(cfg.out, "predictions.jsonl"), test)
    np.save(os.path.join(cfg.out, "oof.npy"), oof_cal)
    gold = load_gold(cfg.test_gold, test)
    if gold:
        report["test_single_runs"] = [M.evaluate(t, gold) for t in tests]
        report["test_weighted"] = M.evaluate(weights @ tests, gold)
        report["test_final"] = M.evaluate(te_cal, gold)
    with open(os.path.join(cfg.out, "metrics.json"), "w") as f:
        json.dump(report, f, indent=2)

    for p, m in zip(cfg.runs, report["single_runs_oof"]):
        print(f"OOF  {os.path.basename(os.path.normpath(p)):>6}: rho={m['spearman']:.4f} acc={m['accuracy']:.4f}")
    print(f"weights={report['weights']} rank_blend={use_blend} calibration s={s:.2f} c={c:+.2f}")
    print(f"OOF  ensemble: rho={report['oof_final']['spearman']:.4f} acc={report['oof_final']['accuracy']:.4f}")
    if gold:
        for p, m in zip(cfg.runs, report["test_single_runs"]):
            print(f"TEST {os.path.basename(os.path.normpath(p)):>6}: rho={m['spearman']:.4f} acc={m['accuracy']:.4f}")
        m = report["test_final"]
        print(f"TEST ensemble: rho={m['spearman']:.4f} acc={m['accuracy']:.4f}")


if __name__ == "__main__":
    main()
