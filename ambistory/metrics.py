"""Official SemEval 2026 Task 5 metrics, reproduced exactly.

Source: github.com/Janosch-Gehring/semeval26-05-scripts/blob/main/scoring.py

    (avg - stdev) < pred < (avg + stdev)   or   |avg - pred| < 1

which is |pred - avg| < max(stdev, 1) with a STRICT inequality. The notebooks
used `<=`, which counts boundary cases (e.g. integer predictions exactly 1.0
away) as correct when the official scorer does not.
"""
import numpy as np


def _rank(x):
    """Average ranks for ties (same as scipy.stats.rankdata)."""
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x))
    sorted_x = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return ranks


def spearman(preds, targets):
    rp, rt = _rank(preds), _rank(targets)
    rp, rt = rp - rp.mean(), rt - rt.mean()
    denom = np.sqrt((rp ** 2).sum() * (rt ** 2).sum())
    return float((rp * rt).sum() / denom) if denom > 0 else float("nan")


def within_sd(preds, averages, stdevs):
    """Boolean array: is each prediction counted correct by the official scorer?"""
    preds, averages, stdevs = (np.asarray(a, dtype=float) for a in (preds, averages, stdevs))
    return np.abs(preds - averages) < np.maximum(stdevs, 1.0)


def accuracy_within_sd(preds, averages, stdevs):
    return float(within_sd(preds, averages, stdevs).mean())


def evaluate(preds, rows):
    """preds aligned with rows (which must carry 'average' and 'stdev')."""
    avg = [r["average"] for r in rows]
    std = [r["stdev"] for r in rows]
    return {"spearman": spearman(preds, avg), "accuracy": accuracy_within_sd(preds, avg, std)}


def summarize_folds(per_fold):
    """per_fold: list of {'spearman', 'accuracy'} -> mean and std, for honest reporting."""
    out = {}
    for key in ("spearman", "accuracy"):
        vals = np.array([m[key] for m in per_fold])
        out[key] = (float(vals.mean()), float(vals.std()))
    return out
