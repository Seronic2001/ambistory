"""Sanity checks for the shared utilities and reproductions of the bugs in REVIEW.md.

Usage (CPU only, needs numpy + torch):
    python tests/sanity_checks.py --data data/

where data/ holds train.json, dev.json and test.json.
"""
import argparse
import os
import random
import statistics
import sys
from collections import Counter

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ambistory import data, metrics, losses, submission  # noqa: E402
from ambistory.batching import StoryBatchSampler  # noqa: E402


def official_is_within_sd(prediction, labels):
    """Verbatim logic of scoring.py::is_within_standard_deviation."""
    avg = sum(labels) / len(labels)
    stdev = statistics.stdev(labels)
    if (avg - stdev) < prediction < (avg + stdev):
        return True
    if abs(avg - prediction) < 1:
        return True
    return False


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    return ok


def main(data_dir):
    train = data.load_split(os.path.join(data_dir, "train.json"))
    dev = data.load_split(os.path.join(data_dir, "dev.json"))
    test = data.load_split(os.path.join(data_dir, "test.json"))
    traindev = train + dev
    results = []

    # --- data structure -------------------------------------------------------
    per_setup = Counter(Counter(data.setup_id(r) for r in traindev).values())
    per_story = Counter(Counter(data.story_id(r) for r in traindev).values())
    results.append(check("every setup has exactly 2 rows", set(per_setup) == {2}, dict(per_setup)))
    results.append(check("every story has exactly 6 rows", set(per_story) == {6}, dict(per_story)))
    results.append(check("uids unique after merging train+dev",
                         len({r["uid"] for r in traindev}) == len(traindev)))

    seen_hom = {r["homonym"] for r in traindev}
    test_hom = {r["homonym"] for r in test}
    overlap = len(seen_hom & test_hom)
    results.append(check("test homonyms are (almost) all unseen", overlap <= 0.05 * len(test_hom),
                         f"{overlap}/{len(test_hom)} test homonyms appear in train+dev"))

    # --- Bug C1: the 'nonsensical' meta feature (Models 7-9) -----------------
    import json
    with open(os.path.join(data_dir, "test.json"), encoding="utf8") as f:
        raw_test = json.load(f)
    old_feat = lambda item: min(sum(1 for v in item.get("nonsensical", []) if v) / 5.0, 1.0)
    test_vals = [old_feat(i) for i in raw_test.values()]
    train_vals = [old_feat(r) for r in train]
    print(f"       old nons feature: train mean={np.mean(train_vals):.3f}, "
          f"train max={max(train_vals):.1f}, test mean={np.mean(test_vals):.3f}")
    results.append(check("C1 reproduced: old feature is 1.0 on every test row ('(???)' has 5 chars)",
                         all(v == 1.0 for v in test_vals)))
    results.append(check("fix: load_split turns test label placeholders into None",
                         all(r["nonsensical"] is None and r["average"] is None for r in test)))

    # --- folds -------------------------------------------------------------------
    fold_of = data.homonym_folds(traindev, 5)
    leaks = 0
    for key_fn in (lambda r: r["homonym"], data.story_id, data.setup_id):
        folds_per_key = {}
        for r, k in zip(traindev, fold_of):
            folds_per_key.setdefault(key_fn(r), set()).add(k)
        leaks += sum(len(v) > 1 for v in folds_per_key.values())
    sizes = Counter(fold_of)
    results.append(check("homonym folds: no homonym/story/setup crosses folds", leaks == 0,
                         f"fold sizes {dict(sorted(sizes.items()))}"))

    # Bug C2: the setup-grouped split of Models 7-9 lets a story's other endings leak
    rng = random.Random(42)
    setups = sorted({data.setup_id(r) for r in train})
    val_setups = set(rng.sample(setups, int(0.15 * len(setups))))
    val_rows = [r for r in train if data.setup_id(r) in val_setups]
    tr_stories = {data.story_id(r) for r in train if data.setup_id(r) not in val_setups}
    tr_homs = {r["homonym"] for r in train if data.setup_id(r) not in val_setups}
    frac_story = np.mean([data.story_id(r) in tr_stories for r in val_rows])
    frac_hom = np.mean([r["homonym"] in tr_homs for r in val_rows])
    results.append(check("C2 reproduced: setup-grouped split leaks stories into validation",
                         frac_story > 0.5,
                         f"{frac_story:.0%} of val rows have their story in train, "
                         f"{frac_hom:.0%} their homonym (test: {overlap / len(test_hom):.0%})"))

    # --- metric ----------------------------------------------------------------------
    rng_np = np.random.default_rng(0)
    preds = np.concatenate([rng_np.uniform(1, 5, len(train)),
                            [r["average"] + 1.0 for r in train[:200]]])  # boundary cases
    rows = train + train[:200]
    ours = metrics.within_sd(preds, [r["average"] for r in rows], [r["stdev"] for r in rows])
    official = np.array([official_is_within_sd(p, r["choices"]) for p, r in zip(preds, rows)])
    notebook = np.abs(preds - np.array([r["average"] for r in rows])) <= np.maximum(
        [r["stdev"] for r in rows], 1.0)
    results.append(check("metrics.within_sd matches official scorer exactly", (ours == official).all()))
    print(f"       notebook '<=' version disagrees with the official scorer on "
          f"{int((notebook != official).sum())} of {len(rows)} rows in this probe")
    sp_ref = metrics.spearman([1, 2, 3, 4, 5, 5], [5, 6, 7, 8, 7, 9])  # by hand: 14.25 / 17
    results.append(check("spearman with ties", abs(sp_ref - 14.25 / 17) < 1e-9, f"{sp_ref:.4f}"))

    # --- Bug H1: ranking loss activity, random vs story batches -----------------------------
    setup_idx = {s: i for i, s in enumerate(sorted({data.setup_id(r) for r in traindev}))}
    gids = torch.tensor([setup_idx[data.setup_id(r)] for r in traindev])
    y = torch.tensor([r["average"] for r in traindev], dtype=torch.float32)
    p = torch.zeros(len(traindev), requires_grad=True)

    def pairs_per_epoch(batches):
        return sum(losses.within_setup_ranking_loss(p[b], y[b], gids[b])[1] for b in batches)

    perm = torch.randperm(len(traindev), generator=torch.Generator().manual_seed(0)).tolist()
    random_batches = [perm[i:i + 32] for i in range(0, len(perm), 32)]
    story_batches = list(StoryBatchSampler([data.story_id(r) for r in traindev], 5, seed=0))
    all_pairs = pairs_per_epoch([list(range(len(traindev)))])
    rand_pairs, story_pairs = pairs_per_epoch(random_batches), pairs_per_epoch(story_batches)
    print(f"       ranking pairs seen per epoch: random batches={rand_pairs}, "
          f"story batches={story_pairs}, available={all_pairs}")
    results.append(check("H1 reproduced: random batches see <5% of ranking pairs",
                         rand_pairs < 0.05 * all_pairs))
    results.append(check("fix: StoryBatchSampler sees every ranking pair", story_pairs == all_pairs))
    covered = sorted(i for b in story_batches for i in b)
    results.append(check("StoryBatchSampler covers each row exactly once",
                         covered == list(range(len(traindev)))))

    # --- Bug H2: Model 6 ranking loss can never be satisfied for small label gaps -------------
    def model6_loss(pr, lab, margin=1.0):
        ld, pd = lab[:, None] - lab[None, :], pr[:, None] - pr[None, :]
        mask = ld.abs() > 0
        return (torch.clamp(margin - ld * pd, min=0) * mask).sum() / mask.sum()
    lab = torch.tensor([3.0, 3.2])
    perfect = torch.tensor([3.0, 3.2])  # exactly right
    results.append(check("H2 reproduced: Model 6 loss penalises a perfect prediction",
                         model6_loss(perfect, lab) > 0.9, f"loss={model6_loss(perfect, lab):.3f}"))
    fixed, _ = losses.within_setup_ranking_loss(torch.tensor([3.0, 3.6]), lab,
                                                torch.tensor([0, 0]), margin=0.3)
    results.append(check("fix: sign-based loss is zero once order and margin are right",
                         float(fixed) == 0.0))

    # --- submission format ------------------------------------------------------------------
    out = os.path.join(os.path.dirname(__file__), "_tmp_predictions.jsonl")
    submission.write_predictions([r["id"] for r in test], np.full(len(test), 3.2), out)
    results.append(check("submission jsonl passes official-style format check",
                         submission.check_against(out, test)))
    os.remove(out)

    print(f"\n{sum(results)}/{len(results)} checks passed")
    return all(results)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(os.path.dirname(__file__), "..", "data"))
    sys.exit(0 if main(ap.parse_args().data) else 1)
