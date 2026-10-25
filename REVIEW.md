# Review: correctness problems and improvements

Every finding below was checked against the notebook code. Items marked 🔬 are also
reproduced by `tests/sanity_checks.py` on the public data.

Severity: **Critical** = invalidates reported results · **High** = silently disables
a component or biases the model · **Medium** = wrong claim, reproducibility or reporting issue.

---

## Critical

### C1. Label-derived input feature `nonsensical` (E07, E08, E09) 🔬
```python
nons_count = sum(1 for v in item["nonsensical"] if v)   # annotator output = part of the label
meta = [clip(nons_count / 5, 0, 1), ending_flag]
```
* `nonsensical` is annotator output. Using it as an input is target leakage (on train it has
  ρ = −0.14 with `average`).
* On the public `test.json` the field is the string `"(???)"`. Iterating a 5-character string
  gives 5 truthy values, so **every test row gets feature = 1.0**. In training the feature
  averages 0.024 and never exceeds 0.6. Every test input is therefore far outside the training distribution.
* This is the most likely main cause of **E09: validation ρ 0.618 → test ρ 0.127**.
* **Measured** (ablation `a12-nons-meta`, adding the feature to fixed E12): OOF ρ +0.013, test ρ −0.004,
  test accuracy **−0.046**. The leak is real, but in an MLP on strong features a feature that is constant at test
  mostly shifts predictions, which ρ ignores. It does not by itself reproduce a collapse like E09's, so "main cause"
  above is an upper bound: E09's RNN presumably relied on the feature much more heavily (not tested).

**Fix:** never build features from `choices / average / stdev / nonsensical`. `ambistory.data.load_split`
now turns the placeholders into `None`, so this class of bug crashes instead of silently producing
values.

### C2. Validation split leaks stories and homonyms (E01–E09) 🔬
* E01–E06: `train_test_split` over rows. Each story has 6 rows, so nearly every validation story
  also appears in training.
* E07–E09: `GroupShuffleSplit(groups=setup_id)`, but `setup_id` includes the ending. The other two
  ending variants of a validation story stay in training: **98% of validation rows have their story
  in train, and 100% have their homonym**. On test, only **1 of 87** homonyms has been seen.
* The homonym embedding in E07–E09 can therefore memorise per-word biases that help validation
  and do nothing on test, where unseen homonyms map to an untrained embedding row.

**Fix:** homonym-grouped K-fold (`ambistory.data.homonym_folds`), as E10–E12 already do. Freeze the folds
to disk (`save_folds`) and use the same split in every experiment.

### C3. Per-homonym mean feature in the final system (E12, also in `app.py`)
```python
homonym_means = df_tr.groupby('homonym')['average'].mean()   # includes each row's own label
vals = [homonym_means.get(h, 3.0) for h in df['homonym']]
```
* In training this includes each row's own label (target leakage), so the MLP learns to rely on it.
* In validation (homonym-grouped) and on test (1/87 homonyms seen) it is the constant **3.0**.
  The feature carries signal during training and none at prediction time. At best it is dead
  weight; at worst it distorts what the MLP learned.

**Fix:** remove it. If you want a homonym prior, use out-of-fold target encoding, and even then it
cannot help on test.
**Measured** (`a12-hom-mean`): OOF ρ −0.005, test ρ −0.001, so it is dead weight, not a large distortion.

### C4. Hyper-parameters and checkpoints selected on the split that is reported
* E01–E02: Keras Tuner objective is `val_mae` on the reported validation set. E03 uses grid search on it.
* E07–E09: 35 Optuna trials plus early stopping on the same validation split, followed by a final
  run on the same split.
* E10–E12: the MLP's best epoch is picked by **outer-fold** Spearman, so the OOF numbers are
  optimistic. E10/E11's Optuna split overlaps the CV folds.
* Under homonym-grouped CV the bias turned out small (E12 OOF 0.543 vs test 0.535). Under the
  leaky splits it compounded C1 and C2.

**Fix:** nested selection. Pick epochs and hyper-parameters on an inner split of the fold-train data
(E12 already does this for the encoder; do it for the MLP too). The outer fold is touched only once.
**Measured** (`a12-select-on-outer`): picking the MLP epoch on the outer fold raises OOF ρ by +0.022 and
accuracy by +0.026, while test ρ changes by −0.005. That is the size of the optimism.

---

## High

### H1. Ranking losses almost never fire (E10, E11, E12) 🔬
E10–E12 add `pairwise_ranking_loss(..., setup_ids)`, but batches are `torch.randperm` slices of
16–32 rows. A setup has exactly 2 rows, so a row's partner is in the same batch only ~1% of the time.
Measured: **random batches of 32 contain 10 of the 1,389 available ranking pairs per epoch (0.7%)**.
The `pairwise_w` and `ft_ranking_w` terms in E10–E12 had essentially no effect.

**Fix:** `ambistory.batching.StoryBatchSampler` puts all 6 rows of a story in one batch, so 100% of
pairs are seen. `losses.within_setup_ranking_loss` returns `n_pairs`; log it.
(E07–E09 used a `SetupBatchSampler` and were fine on this point.)
**Measured:** going back to random batches costs fixed E12 −0.011 OOF / −0.023 test ρ (small but consistent).
In E13 it makes no difference (+0.020 OOF / +0.001 test), and removing E13's ranking term entirely is also neutral.

### H2. E06 ranking loss formula 🔬
```python
loss = clamp(margin - label_diff * pred_diff, min=0)      # over ALL pairs in the batch
```
For two rows with labels 3.0 and 3.2, the *perfect* prediction still gets loss 0.96, because
satisfying the margin needs a predicted gap of 5.0. The loss pushes predictions apart without bound
and fights the MSE term. It also ranks across unrelated stories, which the task does not ask for.
This explains E06 being worse than E05.

**Fix:** `relu(margin − sign(Δy)·Δp)` within setups (as E07+ do; `losses.within_setup_ranking_loss`).

### H3. E09's "contrastive sense loss" duplicates the ranking loss
A setup (story + ending) has exactly 2 rows with different senses, so every within-setup pair
is already a "cross-sense" pair. The comment claims "same-sense pairs (differing in ending) are
skipped". Those pairs never appear, because the ending is part of the setup key. The extra term only
re-weights the ranking loss, and Optuna tuned two redundant hyper-parameters for it.

To compare *across ending conditions* (the same sense should score higher under the ending that
supports it), group by `story_id` instead. That is a genuinely new signal.

### H4. Unmasked padding in all RNN models (E01–E09)
* `padding='post'` without masking or `pack_padded_sequence`. E01 reads the forward RNN state after
  ~140 pad tokens.
* In E02–E08 the meaning (~9 words) is padded to 150, so mean/max pooling and attention run over
  ~94% padding.
* E09 masks attention keys but still mean-pools the padded *query* rows.

Much of the stage 1–2 stagnation is plausibly caused by this rather than by model capacity.

### H5. Stacked features fitted on the encoder's own training rows (E10, E11, E12)
The encoder is fine-tuned on ~85% of fold-train. The MLP is then trained on encoder features of
*those same rows*, where e.g. cosine similarity is over-fitted and looks more predictive than it will
on validation or test. The MLP learns to over-trust it.

**Fix:** cross-fitting (inner K-fold: compute each fold-train row's features with an encoder that did
not train on it), or train the MLP only on the inner held-out 15%, or freeze the encoder.

### H6. E04 applies the weight twice
`WeightedRandomSampler(1/σ²)` **and** a loss weighted by `1/σ²`, so the effective weight is 1/σ⁴.
With `clamp(σ, min=1)` only rows with σ>1 are affected, but they are down-weighted much more than intended.

---

## Medium

### M1. "NLI warm-start" was never ablated (E10 vs E11)
Both notebooks load `cross-encoder/nli-deberta-v3-base`. E10's markdown says
"upgraded to microsoft/deberta-v3-base", but the code doesn't do that. What actually differs:
* E10: linear head during fine-tuning, no PCA.
* E11: MLP head, PCA-128.
* Both: LR (3e-5 vs 2e-5) and NaN handling.

The test difference (0.392 vs 0.400) is smaller than the fold std (±0.04–0.05). The original README's
claim that NLI initialisation helped is unsupported.
**Measured** (`a13-nli-base` vs `a13-base-size`, the same E13 recipe with only the checkpoint changed):
`cross-encoder/nli-deberta-v3-base` scores OOF ρ 0.329 / test 0.384, while `microsoft/deberta-v3-base` scores
0.423 / 0.516. **NLI initialisation hurts.**

### M2. Notebooks don't run top-to-bottom
* E10 calls `fold_pcas.append(pca)`, but `pca` is no longer defined in that notebook. It ran only
  because of leftover kernel state.
* E10 builds differential-LR `param_groups`, then discards them for one AdamW at 2e-5, so `head_lr`
  and `mlp_hidden*` are dead config.
* E11 has `val_acc, _ = compute_metrics(...)` with the return values swapped, so the logged
  "accuracy" during fine-tuning is Spearman.

### M3. Confounded comparisons
* E02→E03 changes framework, tokenizer (`.split()` keeps punctuation), embedding trainability and
  search procedure at once.
* E03→E04 removes attention and HP search while changing the loss.
* E05 also changes the LR.

None of these deltas can be attributed to the change named in the title. Change one thing per experiment.

### M4. Differences reported without noise estimates
* A single 15% split (n≈342) has a Spearman standard error of ≈0.05.
* E12's per-fold ρ ranges 0.42–0.64 (std 0.089).
* E04 vs E05 (0.237 vs 0.245) and E10 vs E11 are ties.

Report mean ± std over folds and ≥3 seeds.

### M5. Metric and submission format
* The notebooks use `|p−avg| <= max(σ,1)`; the official scorer uses strict `<`. In a probe with
  boundary cases the two disagreed on 98 of 2,480 rows 🔬. Use `ambistory.metrics`.
* E10–E12 write `submission.csv` with columns `id,average`, and E09 writes one JSON dict. CodaBench
  expects JSONL lines `{"id": ..., "prediction": ...}`. Use `ambistory.submission.write_predictions`.

### M6. E12 soft-rank blend was never validated
`soft_rank_blend` (pushes the higher sense of each pair toward 5, the lower toward 1, weight 0.15)
is applied to test predictions only and never measured on OOF. Evaluate it on OOF before keeping it.

### M7. `app.py` feeds the model out-of-distribution features
* For a single sample it sets `within_setup_delta = ws_rank = ce_delta = 0`. In training, `ws_rank`
  is always 0 or 1 (two rows per setup), and 0 means "the lower-rated sense of the pair". Single-sample
  scores are therefore biased downward. Score the pair together (the "Compare two meanings" tab should
  call the model on both rows at once).
* It loads `trial1_models.pt`, which is not in the repo, and uses the per-homonym mean from C3.

### M8. Repo hygiene
* The README describes `trial1/`, `trial2/` and `phase2/` folders that don't exist.
* It says E12 uses an "ending flag", but the feature list has none (only `cos_end`, which falls back
  to the ambiguous sentence when there is no ending).
* File names contain stray spaces and en-dashes ("GloVeTrain able", "Interac tions", "Model4–").
* Notebooks hard-code three different data roots (`/content`, two Kaggle paths).
* E01–E08 train on `train.json` only, E09+ on train+dev.

### M9. Minor
* The KL target in E10–E12 drops `nonsensical` votes while `average` keeps them, so the KL and ranking
  terms target slightly different quantities.
* E09's Optuna callback prints `Spearman=nan` for pruned trials (cosmetic).

---

## Improvement roadmap (in priority order)

1. **Freeze one evaluation protocol and re-baseline E12 under it.**
   * Homonym-grouped 5-fold over train+dev (`homonym_folds` + `save_folds`).
   * Nested inner split for every early-stopping and tuning decision.
   * 3 seeds; report OOF mean ± std with `ambistory.metrics`.

   Everything below should be judged against this baseline, one change at a time.

2. **Fix E12 in place.** Each item is small and independently testable:
   * drop the homonym mean (C3);
   * add `StoryBatchSampler` to encoder fine-tuning and MLP training so the ranking losses work (H1);
   * select the MLP epoch on an inner split (C4);
   * cross-fit the MLP features (H5);
   * add a real ending-condition flag (none / has ending);
   * validate or remove `soft_rank_blend` (M6);
   * write JSONL (M5).

3. **Model the pair, not the row.** The two senses of a setup differ by 1.83 points on average.
   * Predict both senses jointly, e.g. a shared encoder followed by a head over
     `[f(A), f(B), f(A)−f(B)]`, or add `CE(A) − CE(B)` and `cos(A) − cos(B)` as features (E12's
     deltas are halves of these).
   * Rank across ending conditions within a story (H3): the sense supported by ending A should
     score higher under A than under "no ending".

4. **Retry the cross-encoder fairly before concluding bi-encoders win.** E10/E11 had inactive ranking
   losses, no example sentence in the input and no homonym marking. Try:
   * story-batched training;
   * input `story [SEP] meaning. Example: …` with the homonym marked;
   * a regression head on `average`;
   * DeBERTa-v3-large if memory allows.

   Then ensemble its OOF predictions with E12 (simple averaging of ranks).

5. **Optimise accuracy without touching Spearman.** Any strictly increasing transform of the
   predictions leaves Spearman unchanged. Fit an isotonic or piecewise-linear calibration on OOF
   predictions to maximise accuracy-within-SD. Predictions here are compressed toward the mean
   (E10's test predictions span only 1.83–4.36), and a monotone stretch costs no Spearman.

6. **Optional: an LLM judge as a feature.** Zero- or few-shot "rate 1–5 how plausible this sense is"
   scores from an instruction-tuned LLM, averaged over a few prompts, used as one more feature or
   ensemble member. Evaluate it under the same folds.

7. **For the report:** present the progression as the *evaluation-protocol* story in README §"The story
   in four lessons". The honest headline is that homonym-grouped CV made validation predictive of test,
   and stage 3's apparent gains came from leakage. That is a stronger and more defensible narrative than
   twelve architecture tweaks.
