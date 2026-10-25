"""Write predictions in the official CodaBench format.

One JSON object per line: {"id": "<key in test.json>", "prediction": <1..5>}.
The notebooks wrote CSV (id,average) or a single JSON dict, and neither passes
format_check.py.
"""
import json

import numpy as np


def write_predictions(ids, preds, path, clip=True):
    ids = [str(i) for i in ids]
    preds = np.asarray(preds, dtype=float)
    if len(ids) != len(preds):
        raise ValueError(f"{len(ids)} ids but {len(preds)} predictions")
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate ids")
    if not np.isfinite(preds).all():
        raise ValueError("non-finite predictions")
    if clip:
        preds = np.clip(preds, 1.0, 5.0)
    with open(path, "w", encoding="utf8") as f:
        for i, p in zip(ids, preds):
            f.write(json.dumps({"id": i, "prediction": float(p)}) + "\n")


def check_against(path, rows):
    """Same checks as the official format_check.py: every id exactly once, values in 1..5."""
    expected = {r["id"] for r in rows}
    seen = set()
    with open(path, encoding="utf8") as f:
        for line in f:
            obj = json.loads(line)
            i, p = str(obj["id"]), float(obj["prediction"])
            if i not in expected or i in seen:
                raise ValueError(f"unexpected or duplicate id {i}")
            if not 1 <= p <= 5:
                raise ValueError(f"prediction out of range for id {i}: {p}")
            seen.add(i)
    missing = expected - seen
    if missing:
        raise ValueError(f"{len(missing)} ids missing, e.g. {sorted(missing)[:5]}")
    return True
