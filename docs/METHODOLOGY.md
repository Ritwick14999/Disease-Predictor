# Methodology

Why this project reports the numbers it reports.

---

## 1. The problem with the headline 100%

The original version of this project trained XGBoost on the symptom dataset and
measured:

```
Train accuracy: 1.0
Test accuracy : 1.0
CV scores: [1. 1. 1. 1. 1.]   Mean CV: 1.0   Std CV: 0.0
```

A perfect score on held-out data, reproduced under 5-fold cross-validation, is
not a success signal. It is a prompt to ask what the evaluation is actually
measuring.

Two structural facts about the dataset explain it.

**Rows repeat.** The table contains far fewer distinct symptom vectors than
rows. A stratified random split shuffles rows, not patterns, so identical rows
land in both the training and the test set. Every such test row is one the model
has already seen, and reproducing it is memorisation being scored as
generalisation.

**The mapping is near-deterministic.** Within the dataset's vocabulary, a given
symptom vector almost always corresponds to exactly one disease. There is
essentially no label noise and no overlap to resolve — the irreducible error is
close to zero, so the ceiling really is 100%.

Together these mean the reported accuracy is a property of the *dataset*, not an
achievement of the *model*. Any classifier with enough capacity to memorise
reaches the same number.

## 2. Measuring it: `dp audit`

```bash
dp audit
```

Reports, from your copy of the file:

| Field | What it tells you |
| --- | --- |
| `duplicate_row_fraction` | Share of rows that exactly repeat an earlier row |
| `n_unique_patterns` | Distinct symptom vectors — the effective sample size |
| `rows_per_unique_pattern` | How many times the average pattern is repeated |
| `n_ambiguous_patterns` | Vectors appearing under more than one disease — the error floor |
| `leakage_risk` | Share of rows having a twin elsewhere, i.e. how much of a random test split is free |

## 3. Fixing it: the grouped split protocol

The fix is to make the split respect patterns rather than rows.

Every row is assigned a **group id** by hashing its symptom vector
(`data.pattern_groups`). Splitting is then done with scikit-learn's
`StratifiedGroupKFold`, which keeps all rows of a group on one side of the
boundary while still balancing classes across folds.

- `--split random` — the textbook protocol. Reproduces the inflated number.
- `--split grouped` — **the default**. Test rows are symptom patterns the model
  has never seen.

`dp train` runs both on the *raw* data and reports the difference as
`leakage_gap`. Deduplicating first and then comparing protocols would measure
nothing, because deduplication is precisely what removes the leak.

Note that both remain optimistic in an absolute sense: they measure
generalisation to unseen *combinations* of a fixed symptom vocabulary, drawn
from one constructed table. Neither is evidence about real patients.

## 4. Baselines, so the model has to earn its complexity

A number means nothing without something to compare it to, so `dp benchmark`
runs every model under the same honest protocol:

| Model | Why it is in the comparison |
| --- | --- |
| `majority` | The floor: predict the most frequent class |
| `rules` | `SymptomSignatureMatcher` — Jaccard overlap against each disease's symptom signature. No gradients, no training loop |
| `tree` | A depth-limited decision tree: how many yes/no questions the task really needs |
| `logreg` | Linear reference, and the interpretability baseline |
| `random_forest`, `xgboost` | The heavy hitters |

The `rules` baseline is the one that matters. If a model with no learned
parameters keeps pace with gradient boosting, the honest conclusion is that the
task is a lookup problem and the ensemble is not earning its complexity. The
benchmark reports `fit_seconds` alongside accuracy so that trade-off is visible
rather than implied.

## 5. Resampling: why the default is `none`

The original pipeline applied SMOTE. Three problems:

1. **It is a no-op here.** The dataset holds exactly 120 rows per disease, so
   there is no imbalance to correct. This is visible in the original notebook
   output, where the "balanced" training set had exactly the same shape as its
   input.
2. **SMOTE is the wrong tool for binary features.** It interpolates between
   neighbours, producing values like 0.4 for indicators that can only be 0 or 1.
   The model then trains on fractional symptoms that cannot occur at inference —
   a train/serve mismatch. `--sampler random` (plain oversampling, which copies
   real rows) is the safe choice when a genuine imbalance needs correcting.
3. **Resampling outside cross-validation leaks.** Synthetic rows built from
   held-out neighbours end up in training folds. This project always places the
   sampler *inside* an `imblearn` pipeline, so it is refit per fold on training
   data only.

## 6. Metrics beyond accuracy

`classification_metrics` always reports:

- **accuracy** — the headline, kept for comparability.
- **balanced accuracy** and **macro-F1** — accuracy hides failure on rare
  classes; these do not.
- **top-3 accuracy** — the realistic criterion for a differential-diagnosis
  shortlist. A list that contains the answer is useful even when its first entry
  is wrong.
- **log loss** and **expected calibration error** — whether the probabilities
  mean anything. A model that is right 95% of the time but always says 99% is
  miscalibrated, and that matters as soon as a person reads the number.

## 7. Robustness: the evaluation that actually distinguishes models

Clean benchmark rows are not the input a real user produces. People forget
symptoms and volunteer irrelevant ones.

`robustness_curve` re-scores the trained model across a grid of two noise types:

- **dropout rate** — probability that a symptom the patient has goes unreported.
- **false-positive rate** — probability that an absent symptom is reported anyway.

Each cell is averaged over several random perturbations of the held-out set.

This is where models that are indistinguishable at 100% separate. Degradation
under symptom dropout is the closest thing this dataset offers to a real-world
performance estimate, and it is the table worth reading first.

## 8. Reproducibility

- Every run is described by one serialisable `TrainingConfig`.
- Seeds are explicit and threaded through splits, models and perturbations.
- The dataset is fingerprinted with SHA-256; two runs reporting the same
  fingerprint saw byte-identical input.
- Library versions are recorded in the bundle and surfaced at `/model-info`.
- The model, the label encoder and the ordered feature list are saved as **one
  bundle**, which makes the classic train/serve column-order mismatch impossible
  rather than merely unlikely.
