# Data directory

Datasets are **not committed to this repository**. `data/raw/` is where the
pipeline looks for them, and it is gitignored.

## Getting the dataset

The project uses the public *Disease Symptom Prediction* table (4,920 rows ×
132 binary symptoms × 41 diseases), distributed on Kaggle as `Training.csv`.

Download it and place it here:

```
data/raw/Training.csv
```

Any of these also work, in this resolution order:

```bash
dp train --data /path/to/Training.csv     # explicit flag wins
export DISEASE_DATA_PATH=/path/to/Training.csv   # then the environment variable
                                          # then data/raw/Training.csv
```

## Running without the dataset

Every command accepts `--synthetic`, which generates a labelled synthetic table
with the same structure — and the same duplicate-row failure mode — so the whole
pipeline can be explored, tested and demonstrated on a machine that has never
seen the real file:

```bash
dp audit --synthetic
dp train --synthetic
```

Anything trained this way is marked `synthetic: true` in its metadata and
reported as such by `/model-info` and the Streamlit sidebar, so a demo run can
never be mistaken for a real one.

## Expected format

A CSV with one binary column per symptom plus a `prognosis` label column:

| itching | skin_rash | ... | prognosis |
| --- | --- | --- | --- |
| 1 | 1 | ... | Fungal infection |
| 0 | 1 | ... | Allergy |

The loader canonicalises headers, drops the unnamed trailing column many copies
of this file carry, and rejects anything non-binary with an explicit error
rather than coercing it.

## A note on real clinical data

If you extend this project with real patient data, that data does not belong in
git regardless of how de-identified it is. Keep it out of the repository and out
of container images.
