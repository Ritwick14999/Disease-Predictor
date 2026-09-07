"""Disease prediction from reported symptoms.

A leakage-aware, reproducible machine-learning pipeline that turns a table of
binary symptom indicators into a calibrated ranking of candidate diagnoses.

The package is deliberately split so that every stage can be tested and reused
on its own:

``config``      Typed configuration objects and default paths.
``data``        Loading, schema validation and duplicate/leakage auditing.
``features``    Turning free-form symptom names into model-ready vectors.
``models``      The model registry, including non-ML reference baselines.
``evaluation``  Metrics, leakage-aware splits and robustness benchmarks.
``explain``     Global and per-prediction attribution.
``training``    The end-to-end training run that produces the model bundle.
``predict``     The inference-time entry point used by the API and the UI.

This project is for education and portfolio purposes. It is **not** a medical
device and must not be used to make health decisions.
"""

from disease_predictor.__about__ import __version__

__all__ = ["__version__"]
