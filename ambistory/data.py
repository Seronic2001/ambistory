"""Loading, grouping and cross-validation folds for AmbiStory (SemEval 2026 Task 5).

Structure of the data (verified on the public train split):
  * a *story* = homonym + precontext + ambiguous sentence      -> 6 rows
  * a *setup* = story + one ending condition (A, B or none)    -> 2 rows (one per sense)
  * only 1 of the 87 test homonyms occurs in train+dev, so the test set is
    effectively "unseen homonyms": validation must group by homonym.
"""
import hashlib
import json
import os
from collections import defaultdict

# Fields produced by the annotators. They are targets, never model inputs.
# In the public test.json they hold the placeholder string "(???)".
LABEL_FIELDS = ("choices", "average", "stdev", "nonsensical")
PLACEHOLDER = "(???)"


def load_split(path, split=None):
    """Return a list of row dicts with 'id' (key in the file), 'split' and 'uid'.

    train.json and dev.json both start at id "0", so use 'uid' ("train:0") as
    the key once splits are merged.

    Placeholder label values are replaced by None, so code that accidentally
    derives features from labels crashes on the test set instead of silently
    producing garbage (e.g. sum(1 for v in "(???)" if v) == 5).
    """
    with open(path, encoding="utf8") as f:
        raw = json.load(f)
    split = split or os.path.splitext(os.path.basename(path))[0]
    rows = []
    for key, item in raw.items():
        row = dict(item)
        row["id"] = str(key)
        row["split"] = split
        row["uid"] = f"{split}:{key}"
        row["ending"] = (row.get("ending") or "").strip()
        for field in LABEL_FIELDS:
            if row.get(field) == PLACEHOLDER:
                row[field] = None
        rows.append(row)
    return rows


def has_labels(rows):
    return all(r.get("average") is not None for r in rows)


def _hash(*parts):
    return hashlib.md5("||".join(parts).encode()).hexdigest()[:12]


def story_id(row):
    """All 6 rows (2 senses x 3 ending conditions) of one story share this id."""
    return _hash(row["homonym"], row["precontext"], row["sentence"])


def setup_id(row):
    """The 2 rows (one per sense) of one story + ending condition share this id."""
    return _hash(row["homonym"], row["precontext"], row["sentence"], row["ending"])


def story_text(row):
    parts = [row["precontext"].strip(), row["sentence"].strip()]
    if row["ending"]:
        parts.append(row["ending"])
    return " ".join(parts)


def meaning_text(row, with_example=True):
    text = row["judged_meaning"].strip()
    if with_example and row.get("example_sentence"):
        text += ". For example: " + row["example_sentence"].strip()
    return text


def ending_condition(row):
    """0 = no ending, 1 = has an ending. Safe input feature (not a label)."""
    return 1 if row["ending"] else 0


def homonym_folds(rows, n_folds=5):
    """Deterministic homonym-grouped K-fold (same greedy rule as sklearn GroupKFold).

    Returns a list of fold indices, one per row. No homonym - and therefore no
    story or setup - is ever split across folds.
    """
    by_hom = defaultdict(list)
    for i, r in enumerate(rows):
        by_hom[r["homonym"]].append(i)
    fold_of = [0] * len(rows)
    fold_sizes = [0] * n_folds
    # largest groups first, each into the currently smallest fold
    for hom in sorted(by_hom, key=lambda h: (-len(by_hom[h]), h)):
        k = min(range(n_folds), key=lambda f: fold_sizes[f])
        for i in by_hom[hom]:
            fold_of[i] = k
        fold_sizes[k] += len(by_hom[hom])
    return fold_of


def save_folds(rows, fold_of, path):
    """Freeze the folds to disk so every experiment uses exactly the same split."""
    with open(path, "w", encoding="utf8") as f:
        json.dump({r["uid"]: k for r, k in zip(rows, fold_of)}, f, indent=0)
