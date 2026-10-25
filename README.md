# AmbiStory — SemEval 2026 Task 5: Rating Plausibility of Word Senses

Given a 5-sentence story containing a homonym, predict how plausible (1–5) humans
find a given sense of that homonym. Scored with **Spearman ρ** against the human mean
and **accuracy within SD**: `|pred − avg| < max(stdev, 1)`
([official scorer](https://github.com/Janosch-Gehring/semeval26-05-scripts/blob/main/scoring.py)).

This repo continues the IIIT Hyderabad INLP course project of Satyajit Priyadarshi and Shubhadeep Mandal
(original team repo, notebooks as submitted: [SATYAJIT2747/INLP-PROJECT](https://github.com/SATYAJIT2747/INLP-PROJECT)).
It reorganises the original 12 notebooks into one progression, adds shared, tested utilities that fix the
correctness problems found in review, and extends the work with ablations, a DeBERTa-v3-large cross-encoder,
zero-shot LLM judges and a deployable app. **Read [REVIEW.md](REVIEW.md) before re-using any number below.**

## Facts about the data that drive every design decision

Verified by `tests/sanity_checks.py` on the public data:

| Fact | Consequence |
|---|---|
| train 2280 / dev 588 / test 930 rows | ~2.9k labelled rows: small, noisy (≥5 annotators/row) |
| each **story** = 6 rows (2 senses × 3 ending conditions: A, B, none) | random row splits put the same story in train and val |
| each **setup** (story + ending) = exactly 2 rows, one per sense | within-setup ranking = "which sense fits better here"; mean sense gap is 1.83 points |
| **1 of 87** test homonyms appears in train+dev | test = unseen homonyms → only **homonym-grouped** CV mimics it; homonym IDs/means are useless at test |
| test label fields are the string `"(???)"` | any feature built from `choices`/`nonsensical`/`stdev` breaks silently on test |

## The progression

The table is split in two. **Part A** holds every model we trained ourselves (stages 1–5, no LLM).
**Part B** adds zero-shot LLM judges (stage 6), alone and blended with the best trained model (stage 7).
★ marks the system picked by OOF ρ in each part.

### Part A: trained models (no LLM)

| ID | Stage | What changed vs. previous | Validation | Val ρ | Val acc | **Test ρ** | **Test acc** |
|---|---|---|---|---|---|---|---|
| E01 | 1 RNN baselines | single BiSimpleRNN on concatenated text | random row split ⚠ | 0.072 | 0.547 | – | – |
| E02 | 1 | dual encoder (story vs. meaning), frozen GloVe, attention | random row split ⚠ | 0.021 | 0.535 | – | – |
| E03 | 1 | GloVe trainable (*and* Keras→PyTorch, new tokenizer, grid search) | random row split ⚠ | 0.186 | 0.570 | – | – |
| E04 | 2 Loss ablations | 1/σ² weighted MSE (*and* attention + HP search removed) | random row split ⚠ | 0.237 | 0.617 | – | – |
| E05 | 2 | Huber δ=1 (*and* LR 1e-3→5e-4) | random row split ⚠ | 0.245 | 0.608 | – | – |
| E06 | 2 | MSE + global pairwise ranking (buggy formula) | random row split ⚠ | 0.174 | 0.541 | – | – |
| E07 | 3 Task-aware RNN | BiGRU, rich interactions, homonym embedding, `nonsensical` meta feature, setup ranking, Optuna | setup-grouped ⚠⚠ | 0.572 | 0.725 | – | – |
| E08 | 3 | + 5-class annotator-distribution KL head | setup-grouped ⚠⚠ | 0.640 | 0.772 | – | – |
| E09 | 3 | Siamese BiGRU + cross-attention + "contrastive" loss, GloVe-300, train+dev, 5 seeds | setup-grouped ⚠⚠ | 0.618 | 0.810 | **0.127** | **0.560** |
| E10 | 4 Transformers | NLI-DeBERTa-v3-base cross-encoder fine-tuned, linear head + MLP on [CLS] | homonym 5-fold OOF | 0.342 | 0.646 | 0.392 | 0.652 |
| E11 | 4 | same backbone, MLP head + PCA-128 on [CLS] | homonym 5-fold OOF | 0.355 | 0.657 | 0.400 | 0.652 |
| E12 | 4 | MPNet bi-encoder fine-tuned per fold + geometric/within-setup features + frozen NLI score + ordinal head + SWA | homonym 5-fold OOF | 0.543 | 0.738 | 0.535 | 0.722 |
| E12-fixed | 5 Fixed + Kaggle | E12 with the REVIEW fixes (`kaggle/e12_train.py`, 2×T4) | homonym 5-fold OOF, inner-split selection | 0.538 | 0.747 | 0.557 | 0.744 |
| E13 | 5 | DeBERTa-v3-large cross-encoder: homonym marked, gloss + example, ordinal head, story batches, LLRD (`kaggle/e13_train.py`), 4 epochs | same | 0.630 | 0.787 | 0.689 | 0.791 |
| E12+E13 | 5 | weighted ensemble + soft-rank blend + monotone calibration, all chosen on OOF (`kaggle/ensemble.py`) | same | 0.649 | 0.800 | 0.686 | 0.812 |
| E13-6ep | 5 | E13 with 6 epochs (now the default) | same | 0.655 | 0.803 | 0.714 | 0.812 |
| E12+E13-6ep | 5 | ensemble with E13-6ep (weights 0.4 / 0.6) | same | 0.667 | 0.813 | 0.702 | 0.834 |
| E12-fixed ×3 | 5 | E12-fixed averaged over seeds 42/43/44 | same | 0.554 | 0.757 | 0.558 | 0.745 |
| E13-6ep ×3 | 5 | E13-6ep averaged over seeds 42/43/44 | same | 0.679 | 0.812 | **0.719** | 0.815 |
| E12+E13 ×3 ★ | 5 | ensemble of the 3-seed runs (weights 0.4 / 0.6), chosen on OOF | same | **0.687** | **0.821** | 0.709 | **0.840** |

Best trained-only system: **E12+E13 ×3** by OOF (test ρ 0.709, acc 0.840). E13-6ep ×3 alone has the highest test ρ (0.719).

### Part B: LLM judges, alone and blended with E13

Nothing is fitted for an LLM judge, so its "validation" scores are simply its scores on train+dev; blends use the
same homonym folds as Part A, with weights, soft-rank blend and calibration chosen on OOF only.

| ID | Stage | What changed vs. previous | Validation | Val ρ | Val acc | **Test ρ** | **Test acc** |
|---|---|---|---|---|---|---|---|
| E14-7B | 6 LLM judges (alone) | zero-shot Qwen2.5-7B-Instruct, expected rating from digit probabilities, 3 prompts incl. one showing both senses (`kaggle/llm_judge.py`), no training | same folds (nothing is fitted) | 0.671 | 0.769 | 0.680 | 0.778 |
| E14-14B | 6 | Qwen2.5-14B-Instruct-AWQ, vLLM tensor-parallel over 2×T4 | same | 0.703 | 0.821 | 0.742 | 0.834 |
| E14-32B | 6 | Qwen2.5-32B-Instruct-AWQ, vLLM tensor-parallel over 2×T4 | same | 0.756 | 0.816 | 0.778 | 0.837 |
| E14-14B+E14-32B | 6 | the two judges only, weights 0.2 / 0.8 | same | 0.758 | 0.872 | 0.784 | 0.885 |
| E13×3+E14-7B | 7 Trained + LLM | E13-6ep ×3 + E14-7B: ensemble (0.55 / 0.45) + soft-rank blend + calibration, chosen on OOF | same | 0.735 | 0.847 | 0.754 | 0.848 |
| E13×3+E14-14B | 7 | E13-6ep ×3 + E14-14B, weights 0.45 / 0.55 | same | 0.753 | 0.867 | 0.791 | 0.865 |
| E13×3+E14-14B+E14-32B | 7 | equal-weight 3-way ensemble | same | 0.778 | 0.880 | 0.810 | **0.901** |
| E13×3+E14-32B ★ | 7 | E13-6ep ×3 + E14-32B, weights 0.45 / 0.55 | same | **0.783** | **0.890** | **0.811** | 0.899 |

⚠ = validation leaks the same story into train and val. ⚠⚠ = additionally uses a label-derived
feature and leaks homonyms. Validation accuracies of E01–E12 use the notebooks' `<=` rule; rows from E12-fixed on use the official metric and, where the soft-rank blend was selected, the blended predictions.
Machine-readable: [`results/scores.csv`](results/scores.csv), [`results/test_scores.json`](results/test_scores.json).

### The story in four lessons

1. **Stage 1–2: the RNNs never learned the task (ρ ≤ 0.25).** Unmasked post-padding (E01's forward RNN
   is read after ~140 pad tokens; in E02–E08 the 9-word meaning is padded to 150, so ~94% of what gets
   mean-pooled is padding), no access to the sense pair, and validation on single random splits
   (n≈342, ρ standard error ≈ 0.05). The loss ablations (E04–E06) differ by less than that noise, and each also
   changed other things, so no conclusion about losses can be drawn from them.
2. **Stage 3: the "breakthrough" was mostly leakage.** E07–E09 jumped to ρ≈0.6 on validation, but E09 scored
   **0.127 on test**. The prime suspect: the `nonsensical` annotator count is a label, and on the
   public test file it evaluates to 1.0 ("every annotator said nonsense") for *every* row (see the ablations for how
   much of the collapse this explains). The setup-grouped split
   also left 98% of validation stories (and 100% of homonyms) visible in training.
   The real lasting contributions of this stage are *ideas*: within-setup ranking and the annotator-distribution head.
3. **Stage 4: switching to homonym-grouped CV made validation honest.** OOF and test now agree
   (E12: 0.543 vs. 0.535; E10/11 within one fold-std). This is the most important methodological step in the project.
4. **Pretrained sentence similarity + explicit pair features beat end-to-end fine-tuning.** E12 beat the fine-tuned
   cross-encoders by ~0.14 ρ. Note however that (a) E10 vs. E11 is *not* an NLI-vs-no-NLI ablation (both use the NLI
   checkpoint), and (b) the ranking losses in E10–E12 were effectively inactive (random batches see 0.7% of pairs),
   so the cross-encoder may have been under-trained rather than inferior.
   **The ablations below overturn this lesson:** the cross-encoder lost because of its NLI checkpoint and its size, not
   because cross-encoders are worse. A plain DeBERTa-v3-large cross-encoder (E13) beats E12 by 0.14 test ρ.

## Which earlier ideas actually helped (ablations)

Each ablation changes **one thing** in fixed E12 or E13 (seed 42, same homonym folds, run as a separate
Kaggle kernel via `kaggle/ablations.py`). Numbers are raw predictions, before any soft-rank blend.
Full table: [`results/ablations/ablations.json`](results/ablations/ablations.json).

**How big is noise?** Re-running the same configuration with seeds 42/43/44 moves raw OOF ρ by up to 0.009
(E12 0.528–0.537, E13 0.613–0.622). The E13 seed-42 reference is the lowest of its three seeds, so small positive
OOF deltas against it are expected by chance. Below, an idea counts as an effect only if the change is larger than
about 0.015 ρ and OOF and test agree on its direction.

| Idea (where it came from) | Ablation | Δ OOF ρ | Δ test ρ | Δ test acc | Verdict |
|---|---|---|---|---|---|
| Per-homonym mean feature (E12, C3) | add it to fixed E12 | −0.005 | −0.001 | +0.002 | **useless**: constant on unseen homonyms |
| `nonsensical` meta feature (E07–E09, C1) | add it to fixed E12 | +0.013 | −0.004 | **−0.046** | **harmful leak**: makes validation look better, costs test accuracy |
| Pick the epoch on the scored fold (E10–E12, C4) | select on outer fold | +0.022 | −0.005 | +0.003 | **optimistic by ≈0.02 ρ**: inflates validation only |
| Story batches, so ranking pairs exist (H1) | random batches in E12 | −0.011 | −0.023 | −0.012 | **useful for E12** (small) |
| SWA (E12) | remove | +0.005 | −0.004 | −0.009 | neutral |
| Frozen NLI entailment score feature (E12) | remove | +0.004 | +0.002 | −0.001 | neutral: can be dropped |
| Fine-tuning MPNet per fold (E12) | freeze encoder | −0.119 | −0.174 | −0.133 | **essential** |
| Ordinal + KL + ranking objective (E07/E08 ideas) | E13 with plain MSE regression | +0.004 | −0.013 | +0.003 | neutral to slightly useful |
| Huber loss (E05) | Huber vs MSE, regression E13 | −0.030 | −0.002 | −0.010 | **not useful** (worse on OOF) |
| 1/σ²-weighted MSE (E04) | vs MSE, regression E13 | −0.001 | +0.009 | +0.016 | neutral |
| Global ranking loss as in E06 | MSE + E06 loss vs MSE | +0.013 | +0.005 | +0.010 | neutral, maybe slightly positive |
| Within-setup ranking loss (E07) | remove from E13 | +0.008 | −0.006 | −0.007 | neutral alongside KL + MSE |
| Story batches (H1) | random batches in E13 | +0.020 | +0.001 | +0.007 | neutral: E13 doesn't need the ranking term |
| Annotator-distribution KL target (E08) | remove from E13 | +0.006 | −0.012 | +0.011 | neutral |
| Example sentence in the sense text (E13) | remove | −0.012 | **−0.030** | −0.016 | **useful** |
| Quoting the homonym in the story (E13) | remove | +0.014 | −0.006 | +0.019 | neutral |
| Backbone size (large vs base) | deberta-v3-base | −0.191 | −0.173 | −0.111 | **the biggest lever** |
| NLI initialisation (E10/E11, M1) | nli-deberta-v3-base vs deberta-v3-base | −0.093 | **−0.132** | −0.028 | **harmful**: the NLI checkpoint was the wrong start |

What this changes in the story:
- **What mattered:** a large pretrained encoder that is fine-tuned on the task (the two largest effects in the table),
  plus giving the model the sense's example sentence.
  Most of the loss engineering from stages 2–3 (Huber, 1/σ² weights, the KL head, ranking losses) is within noise
  once the encoder is strong.
- **NLI initialisation actively hurt.** With the NLI checkpoint, the base cross-encoder reaches OOF ρ 0.33, about
  where E10/E11 landed (0.34/0.36). With plain `deberta-v3-base` it reaches 0.42. E10/E11 were held back by the
  starting checkpoint, which is the opposite of what the original README claimed.
- **The leaks behave as REVIEW predicted, but are smaller here than in E09.** In E12's feature MLP, the
  `nonsensical` feature inflates validation and costs 4.6 points of test accuracy. Test ρ barely moves, because a
  feature that is constant across the test set mostly shifts every prediction by the same amount, and ranking ignores
  a uniform shift. So this leak alone does not reproduce E09's collapse to 0.127. That collapse also needed E09's
  RNN, which presumably leaned much harder on the feature (likely, not tested).
- **Selecting on the scored fold** adds about 0.02 to validation ρ and nothing to test, so the nested selection was the
  right fix.

## Layout

```
./
├── README.md             this file: task, data facts, progression
├── app.py                Streamlit scorer serving E13 (see "Running the app")
├── REVIEW.md             correctness problems (ranked) + improvement roadmap
├── experiments/          original notebooks, renamed E01–E12, grouped by stage (outputs preserved)
│   ├── stage1_rnn_baselines/
│   ├── stage2_loss_ablations/
│   ├── stage3_task_aware_rnn/
│   └── stage4_transformers/      E12_..._FINAL.ipynb is the submitted system
├── ambistory/            shared utilities with the fixes applied — import these in new experiments
│   ├── data.py           loading (test label placeholders → None), story/setup ids, homonym folds
│   ├── metrics.py        exact replica of the official scorer (strict <)
│   ├── batching.py       StoryBatchSampler: keeps a story's 6 rows in one batch
│   ├── losses.py         within-setup ranking loss (sign-based, reports #pairs), KL from logits
│   └── submission.py     official JSONL writer + format check
├── kaggle/
│   ├── e12_train.py      E12 with the REVIEW fixes, fp16, one worker per GPU, resumable
│   ├── e13_train.py      DeBERTa-v3-large cross-encoder (same folds, flags for every ablation; --deploy for app.py)
│   ├── llm_judge.py      E14 zero-shot LLM judge (hf or vLLM tensor-parallel backend)
│   ├── ensemble.py       OOF-chosen weights, rank blend and calibration over several runs
│   ├── ablations.py      generate / run / collect the 18 one-change ablation kernels
│   ├── kernels.py        generates the E14 judge and E13 deploy Kaggle kernels
│   └── run_e12_kaggle.ipynb  3-cell launcher for Kaggle "GPU T4 x2"
├── tests/sanity_checks.py   reproduces each bug and verifies each fix (16 checks, CPU, ~10 s)
├── results/              all reported numbers in one place
├── models/e13/           deployed E13 checkpoints for app.py (weights git-ignored, ~870 MB per seed)
└── data/                 put train.json / dev.json / test.json here (git-ignored)
```

The original team repo's `app.py` is the E12 demo: it needs `trial1_models.pt` and the old feature layout
(REVIEW.md §M7). `app.py` here replaces it.

## Running the app

`app.py` serves **E13**, the best trained (non-LLM) model: the DeBERTa-v3-large cross-encoder fitted on all of
train+dev, 2 seeds averaged. On the official test set these exact checkpoints score Spearman **0.702** and accuracy
**80.1%** (`models/e13/deploy_metrics.json`, with the within-setup rank blend; 0.681 / 78.0% without it).

```bash
# 1. weights: nothing to do. If models/e13/ is empty, the app downloads them (~1.7 GB) from
#    https://huggingface.co/Seronic/ambistory-e13. To retrain instead (8 min on Kaggle 2x T4):
python kaggle/kernels.py --dir kaggle/kernels/
kaggle kernels push -p kaggle/kernels/e13-deploy --accelerator NvidiaTeslaT4
kaggle kernels output shubhadeepmandal/ambistory-e13-deploy -p out/   # copy out/e13_deploy/model_seed* to models/e13/
pip install -r requirements.txt                           # tested versions; CPU-only torch: see the file
streamlit run app.py                                      # 2. serve (AMBISTORY_MODELS=... to use another folder)
```

Tabs: score one story + meaning (with the 5-class rating distribution); compare two meanings of the same story
(the rank blend is applied to the pair); batch-score an AmbiStory JSON file, with official metrics when it has labels
and a CodaBench `predictions.jsonl` download. Runs on CPU (~3.5 GB RAM) or GPU (fp16 backbone, fits in 3 GB).
The LLM blends (E14) are not served: they need a 14B–32B model on GPUs.

Deployment notes (measured on the deployed checkpoints):
- **Quantization does not work here.** PyTorch dynamic int8 on every linear layer drops the agreement with the
  full-precision scores to a correlation of ~0.03–0.08 (test ρ 0.70 → 0.13). The culprit is the feed-forward layers
  (DeBERTa-v3's activation outliers): quantizing only those 48 layers already breaks it, while quantizing only the
  attention layers keeps 0.98 correlation but is just 1.17× faster. So the app serves the weights unquantized.
- **CPU cost:** ~1.2 s per story–meaning pair per model on a 4-core laptop, i.e. ~2.3 s per score with both seeds;
  ~3.3 GB RAM (fp32), which is more than Streamlit Community Cloud provides.
- **Hosting:** Hugging Face Docker/Gradio Spaces need a PRO subscription even on the free CPU tier, so there is no
  hosted demo; the app runs locally and fetches the weights from Hugging Face.

## Training E12 on Kaggle (both GPUs)

Open `kaggle/run_e12_kaggle.ipynb` on Kaggle with *GPU T4 x2* and Internet on, attach the data, and run the cells.
Or run the script directly:

```bash
python kaggle/e12_train.py                      # all visible GPUs, 5 folds, seed 42
python kaggle/e12_train.py --seeds 42,43        # 10 jobs: keeps both GPUs busy, 2-seed ensemble
python kaggle/e12_train.py --gpus 0 --fresh     # one GPU, recompute everything
```

How it uses the GPUs:
1. **NLI scores:** the frozen NLI cross-encoder scores are split across GPUs and cached.
2. **Folds:** each fold (× seed) is an independent job, pulled from a queue by one worker process per GPU.
3. **Aggregation:** OOF metrics use the official scorer. The soft-rank blend is applied only if it improves OOF.
   Output is `predictions.jsonl` in CodaBench format, plus `metrics.json`, `oof.npy` and `folds.json`.

Folds are saved as they finish, so re-running after a timeout continues where it stopped.
Note that the original team repo's `app.py` expects the old feature layout (it includes the homonym mean) and won't load these models.

## Running the checks

```bash
# data: https://github.com/Janosch-Gehring/semeval26-05-scripts/tree/main/data
python tests/sanity_checks.py --data data/
```

## Using the utilities in a notebook

```python
import sys; sys.path.append("path/to/ambistory")   # this repo
from ambistory import data, metrics, losses, submission
from ambistory.batching import StoryBatchSampler

rows  = data.load_split("data/train.json") + data.load_split("data/dev.json")
folds = data.homonym_folds(rows, 5)                     # same split for every experiment
sampler = StoryBatchSampler([data.story_id(r) for r in rows_train], stories_per_batch=5)
loss_rank, n_pairs = losses.within_setup_ranking_loss(pred, y, setup_ids)   # log n_pairs!
print(metrics.evaluate(oof_preds, rows))
submission.write_predictions([r["id"] for r in test], test_preds, "predictions.jsonl")
```
