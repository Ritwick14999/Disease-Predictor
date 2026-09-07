# Data card

## Source

The public **Disease Symptom Prediction** table (commonly distributed on Kaggle
as `Training.csv`), a symptom → prognosis lookup where every symptom is a binary
indicator.

The file is **not committed to this repository**. See [`data/README.md`](../data/README.md)
for how to obtain it, and note that `--synthetic` runs the entire pipeline
without it.

## Structure

| Property | Value |
| --- | --- |
| Rows | 4,920 |
| Feature columns | 132 binary symptom indicators |
| Target column | `prognosis` (string disease name) |
| Classes | 41 |
| Missing values | 0 |
| Rows per class | 120, for every class |

`dp audit` regenerates every number below from your copy of the file, so nothing
here has to be taken on trust.

## Known defects

These are properties of the source data, not bugs in this project. Each one
changes how a result should be read, which is why the pipeline measures them
instead of ignoring them.

### 1. Repeated rows

Distinct symptom vectors are far fewer than rows. The consequence is specific
and severe: **a random train/test split puts byte-identical rows on both sides**,
so a model that memorises scores near-perfectly without generalising at all.

The pipeline handles this in two ways — `dp audit` quantifies it, and the
default `grouped` split protocol keeps every copy of a pattern on one side of
the boundary. See [`METHODOLOGY.md`](METHODOLOGY.md).

### 2. Exact class balance

4,920 ÷ 41 = **exactly 120 rows per disease**, and an 80/20 stratified split
yields exactly 24 test rows per disease. Real prevalence is nothing like this.

Two consequences:

- Class-imbalance handling is **unnecessary here**. Running SMOTE on this table
  is a no-op that changes no row counts — visible in the original notebook,
  where the "balanced" training set had exactly the same shape as its input.
- Accuracy and macro-F1 coincide, so accuracy looks better-behaved than it would
  be on any realistic prevalence distribution.

### 3. The mapping is close to deterministic

Symptom vectors rarely map to more than one disease, so the Bayes error is near
zero *within the dataset's own vocabulary*. The task, as posed, is much closer
to a lookup table than to diagnosis. `dp audit` reports this as
`n_ambiguous_patterns`.

### 4. Label and header noise

The raw file contains misspelled labels (`Peptic ulcer diseae`,
`(vertigo) Paroymsal  Positional Vertigo`), a trailing space in `Diabetes `, and
inconsistent headers (`dischromic _patches`, `foul_smell_of urine`). Labels are
preserved verbatim so results stay comparable with other work on this dataset;
headers are canonicalised on load by `canonical_symptom`.

### 5. No provenance or patient population

The dataset has no documented collection method, geography, age distribution or
time period. There is no basis for claiming the symptom-disease relationships
generalise to any real population.

## Collection and consent

Unknown. The data appears to be constructed from symptom-disease reference
tables rather than collected from patients. It should not be treated as clinical
evidence.

## Recommended use

Learning and benchmarking of multi-class classification and evaluation
methodology. **Not** a basis for any medical claim.
