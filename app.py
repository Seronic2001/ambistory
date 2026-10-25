"""AmbiStory plausibility scorer (Streamlit) serving the best trained system: E13.

E13 = DeBERTa-v3-large cross-encoder with an ordinal head, fitted on all of train+dev
(`kaggle/e13_train.py --deploy`), one model per seed, predictions averaged over seeds. When both
senses of a setup are scored together (Compare tab, batch upload) the within-setup soft-rank blend
used in the evaluated system is applied as well.

    streamlit run app.py        # models from models/e13/model_seed*, or set AMBISTORY_MODELS
"""
import glob
import io
import json
import os
import sys

import numpy as np
import pandas as pd
import streamlit as st
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "kaggle"))
from ambistory import data as D  # noqa: E402
from ambistory import metrics as M  # noqa: E402
from e12_train import soft_rank_blend  # noqa: E402
from e13_train import build_inputs, load_deployed  # noqa: E402

MODEL_DIR = os.environ.get("AMBISTORY_MODELS", os.path.join(HERE, "models", "e13"))
HF_REPO = os.environ.get("AMBISTORY_HF_REPO", "Seronic/ambistory-e13")  # downloaded when MODEL_DIR has no models
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH = 8


@st.cache_resource(show_spinner=False)
def load_models():
    dirs = sorted(glob.glob(os.path.join(MODEL_DIR, "model_seed*")))
    if not dirs and HF_REPO:
        from huggingface_hub import snapshot_download
        snapshot_download(HF_REPO, local_dir=MODEL_DIR, allow_patterns=["model_seed*/*", "deploy_metrics.json"])
        dirs = sorted(glob.glob(os.path.join(MODEL_DIR, "model_seed*")))
    if not dirs:
        return [], {}
    models = []
    for d in dirs:
        model, tok, meta = load_deployed(d, DEVICE)
        if DEVICE.type == "cuda":
            model.backbone.half()  # the ordinal head stays fp32, as in training
        models.append((model, tok, meta))
    report = {}
    path = os.path.join(MODEL_DIR, "deploy_metrics.json")
    if os.path.exists(path):
        with open(path) as f:
            report = json.load(f)
    return models, report


@torch.no_grad()
def score_rows(rows, progress=None):
    """Expected rating and 5-class probabilities per row, averaged over the seed models."""
    models, _ = load_models()
    exp = np.zeros(len(rows))
    probs = np.zeros((len(rows), 5))
    steps, done = len(models) * max(1, -(-len(rows) // BATCH)), 0
    for model, tok, meta in models:
        A, B = build_inputs(rows, mark=meta["mark"], example=meta["example"])
        for s in range(0, len(rows), BATCH):
            enc = tok(A[s:s + BATCH], B[s:s + BATCH], padding=True, truncation="longest_first",
                      max_length=meta["max_len"], return_tensors="pt").to(DEVICE)
            p, e = model(enc)
            probs[s:s + BATCH] += p.float().cpu().numpy() / len(models)
            exp[s:s + BATCH] += e.float().cpu().numpy() / len(models)
            done += 1
            if progress:
                progress(done / steps)
    return np.clip(exp, 1.0, 5.0), probs


def blend_setups(preds, rows):
    """Within-setup soft-rank blend (needs both senses of a setup in `rows`)."""
    models, _ = load_models()
    w = models[0][2].get("rank_blend_w", 0.15)
    ids = {}
    gid = np.array([ids.setdefault(D.setup_id(r), len(ids)) for r in rows])
    return soft_rank_blend(np.asarray(preds, dtype=float), gid, w)


def make_row(homonym, precontext, sentence, ending, meaning, example):
    return {"homonym": homonym.strip(), "precontext": precontext.strip(), "sentence": sentence.strip(),
            "ending": (ending or "").strip(), "judged_meaning": meaning.strip(),
            "example_sentence": (example or "").strip()}


def verdict(score):
    if score >= 4.0:
        return "Highly plausible"
    if score >= 3.0:
        return "Plausible"
    if score >= 2.0:
        return "Unlikely"
    return "Implausible"


def show_score(score, probs, label="Plausibility"):
    st.metric(label, f"{score:.2f} / 5", verdict(score), delta_color="off")
    df = pd.DataFrame({"rating": [str(k) for k in range(1, 6)], "probability": probs})
    st.vega_lite_chart(df, {"mark": "bar", "height": 160, "encoding": {
        "x": {"field": "rating", "type": "ordinal", "axis": {"labelAngle": 0}},
        "y": {"field": "probability", "type": "quantitative"}}}, width="stretch")


# ---------------------------------------------------------------------------
st.set_page_config(page_title="AmbiStory Scorer", layout="wide")
st.title("AmbiStory Plausibility Scorer")
st.caption("How plausible is a given sense of an ambiguous word in a short story? (SemEval 2026 Task 5)")

with st.spinner("Loading models (first run downloads ~1.7 GB from Hugging Face if they are not local)..."):
    MODELS, REPORT = load_models()

with st.sidebar:
    st.subheader("Model")
    if not MODELS:
        st.error(f"No models found in `{MODEL_DIR}`. Train them with "
                 "`kaggle/e13_train.py --deploy` and put the `model_seed*` folders there.")
    else:
        st.markdown(f"**E13**: DeBERTa-v3-large cross-encoder, ordinal head, {len(MODELS)} seed(s) averaged")
        if "test_mean_rank_blend" in REPORT:
            m = REPORT["test_mean_rank_blend"]
            st.markdown(f"Official test set: Spearman **{m['spearman']:.3f}**, accuracy **{m['accuracy']:.1%}**")
    st.markdown(f"Device: **{torch.cuda.get_device_name(0) if DEVICE.type == 'cuda' else 'CPU'}**")
    st.caption("Accuracy = prediction within max(annotator SD, 1) of the mean human rating.")

if not MODELS:
    st.stop()

tab1, tab2, tab3 = st.tabs(["Single scenario", "Compare two meanings", "Batch (JSON)"])

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Story")
        h = st.text_input("Homonym", "drive", key="s_h")
        pre = st.text_area("Precontext", "Lisa had always been competitive. Every weekend, she dedicated herself "
                           "to her passion. She believed that her relentless practice would pay off someday.",
                           key="s_p")
        sent = st.text_area("Ambiguous sentence", "Her drive was what ultimately got her into the top university.",
                            key="s_s")
        end = st.text_area("Ending (optional)", "She made that long trip to show the course coordinators her "
                           "dedication, and they said that was one of the reasons why they accepted her.", key="s_e")
    with c2:
        st.subheader("Meaning")
        mean = st.text_area("Meaning to evaluate", "a strong motivation to achieve something", key="s_m")
        ex = st.text_area("Example sentence (optional)", "Her drive to succeed was obvious.", key="s_x")
        if st.button("Score", type="primary", key="s_btn"):
            if not (h.strip() and sent.strip() and mean.strip()):
                st.error("Homonym, ambiguous sentence and meaning are required.")
            else:
                with st.spinner("Scoring..."):
                    pred, probs = score_rows([make_row(h, pre, sent, end, mean, ex)])
                show_score(pred[0], probs[0])

with tab2:
    st.caption("Both meanings are scored against the same story; the soft-rank blend of the evaluated "
               "system is applied to the pair.")
    c0, ca, cb = st.columns([1.2, 1, 1])
    with c0:
        st.subheader("Story")
        h2 = st.text_input("Homonym", "match", key="c_h")
        pre2 = st.text_area("Precontext", "The rain finally stopped, and the crowd began to gather. "
                            "John checked the pockets of his coat nervously.", key="c_p")
        sent2 = st.text_area("Ambiguous sentence", "The match was striking.", key="c_s")
        end2 = st.text_area("Ending (optional)", "But the box was wet, and it wouldn't light.", key="c_e")
    with ca:
        st.subheader("Meaning A")
        ma = st.text_area("Meaning", "lighter consisting of a thin piece of wood tipped with combustible chemical",
                          key="c_ma")
        xa = st.text_area("Example (optional)", "He struck a match to light the candle.", key="c_xa")
    with cb:
        st.subheader("Meaning B")
        mb = st.text_area("Meaning", "a formal contest in which two or more persons or teams compete", key="c_mb")
        xb = st.text_area("Example (optional)", "The teams prepared for the upcoming match.", key="c_xb")
    if st.button("Compare", type="primary", key="c_btn"):
        if not (h2.strip() and sent2.strip() and ma.strip() and mb.strip()):
            st.error("Homonym, ambiguous sentence and both meanings are required.")
        else:
            rows = [make_row(h2, pre2, sent2, end2, ma, xa), make_row(h2, pre2, sent2, end2, mb, xb)]
            with st.spinner("Scoring..."):
                raw, probs = score_rows(rows)
            final = blend_setups(raw, rows)
            ra, rb = st.columns(2)
            with ra:
                show_score(final[0], probs[0], "Meaning A")
            with rb:
                show_score(final[1], probs[1], "Meaning B")
            if abs(raw[0] - raw[1]) < 0.1:
                st.info("Both meanings are about equally plausible here.")
            else:
                st.success(f"Meaning {'A' if raw[0] > raw[1] else 'B'} fits this story better "
                           f"(raw scores {raw[0]:.2f} vs {raw[1]:.2f}).")

with tab3:
    st.markdown("Upload an AmbiStory JSON file (`{id: {homonym, precontext, sentence, ending, judged_meaning, "
                "example_sentence, ...}}`), e.g. `data/dev.json`. If it has labels (`average`, `stdev`), the "
                "official metrics are reported.")
    up = st.file_uploader("AmbiStory JSON", type=["json"])
    if up is not None:
        raw_json = json.load(up)
        rows = []
        for key, item in raw_json.items():
            r = make_row(item.get("homonym", ""), item.get("precontext", ""), item.get("sentence", ""),
                         item.get("ending", ""), item.get("judged_meaning", ""), item.get("example_sentence", ""))
            r["id"] = str(key)
            for f in ("average", "stdev"):
                v = item.get(f)
                r[f] = v if isinstance(v, (int, float)) else None  # test files hold the placeholder "(???)"
            rows.append(r)
        st.write(f"{len(rows)} rows loaded.")
        if st.button("Score all", type="primary", key="b_btn"):
            bar = st.progress(0.0, text="Scoring...")
            raw, _ = score_rows(rows, progress=lambda f: bar.progress(min(f, 1.0), text=f"Scoring... {f:.0%}"))
            bar.empty()
            final = blend_setups(raw, rows)
            df = pd.DataFrame({"id": [r["id"] for r in rows], "homonym": [r["homonym"] for r in rows],
                               "meaning": [r["judged_meaning"] for r in rows],
                               "has ending": [bool(r["ending"]) for r in rows],
                               "prediction": np.round(final, 3), "gold": [r["average"] for r in rows]})
            if D.has_labels(rows):
                m = M.evaluate(final, rows)
                k1, k2 = st.columns(2)
                k1.metric("Spearman ρ", f"{m['spearman']:.4f}")
                k2.metric("Accuracy (within SD)", f"{m['accuracy']:.1%}")
            st.dataframe(df, use_container_width=True, hide_index=True)
            buf = io.StringIO()
            for r, p in zip(rows, final):
                buf.write(json.dumps({"id": r["id"], "prediction": float(p)}) + "\n")
            st.download_button("Download predictions.jsonl (CodaBench format)", buf.getvalue(),
                               file_name="predictions.jsonl", mime="application/json")
