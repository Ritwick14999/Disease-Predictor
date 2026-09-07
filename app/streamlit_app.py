"""Streamlit front end for the disease predictor.

Run with ``streamlit run app/streamlit_app.py``.

The UI is intentionally more than a symptom picker: alongside the ranking it
shows *why* each disease was proposed and which unreported symptom would be most
informative to check next, because a probability with no reasoning attached is
not something a person can sensibly act on -- and the disclaimer is on screen at
all times, not buried in a footer.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Allow `streamlit run app/streamlit_app.py` from a checkout without installing.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from disease_predictor.config import DISCLAIMER  # noqa: E402
from disease_predictor.predict import DiseasePredictor  # noqa: E402

st.set_page_config(page_title="Disease Predictor", page_icon="🩺", layout="wide")


@st.cache_resource(show_spinner="Loading model…")
def load_predictor(model_path: str | None = None) -> DiseasePredictor:
    """Load the model once per session rather than once per interaction."""
    return DiseasePredictor.load(model_path)


def render_sidebar(predictor: DiseasePredictor) -> None:
    """Show the provenance of the artifact answering the user's questions."""
    metadata = predictor.metadata
    config = metadata.get("config", {})
    holdout = metadata.get("holdout", {})
    audit = metadata.get("leakage_audit", {})

    with st.sidebar:
        st.header("Model")
        st.caption(f"Trained {metadata.get('created_at', 'unknown')}")
        st.metric("Model", config.get("model_name", "unknown"))
        st.metric("Diseases", len(predictor.diseases))
        st.metric("Symptoms", len(predictor.symptoms))

        if metadata.get("synthetic"):
            st.warning("This model was trained on **synthetic** data. Output is a demo only.")

        if holdout:
            st.subheader("Held-out scores")
            st.caption(f"Split protocol: `{config.get('split', {}).get('strategy', 'unknown')}`")
            st.dataframe(
                pd.DataFrame(
                    {"metric": list(holdout), "value": [round(v, 4) for v in holdout.values()]}
                ),
                hide_index=True,
                use_container_width=True,
            )

        if audit:
            st.subheader("Dataset leakage audit")
            st.caption(
                f"{float(audit.get('duplicate_row_fraction', 0)):.1%} of the source rows are exact "
                f"duplicates across {audit.get('n_unique_patterns', '?')} unique symptom patterns."
            )


def render_predictions(result) -> None:
    """Render the ranked shortlist with its attribution."""
    for warning in result.warnings:
        st.warning(warning)

    if not result.predictions:
        return

    st.subheader("Most likely conditions")
    columns = st.columns(len(result.predictions))
    for column, prediction in zip(columns, result.predictions, strict=True):
        column.metric(
            f"#{prediction.rank} {prediction.disease}",
            f"{prediction.probability:.1%}",
        )

    st.progress(min(1.0, result.predictions[0].probability))

    for prediction in result.predictions:
        supporting = [c for c in prediction.supporting_symptoms if c.contribution > 0]
        if not supporting:
            continue
        with st.expander(f"Why #{prediction.rank}: {prediction.disease}?"):
            st.caption(
                "Each bar is the probability this condition loses when that symptom is removed."
            )
            st.bar_chart(
                pd.DataFrame(
                    {"symptom": [c.symptom for c in supporting],
                     "evidence": [c.contribution for c in supporting]}
                ).set_index("symptom")
            )

    if result.follow_up_symptoms:
        st.subheader("Most informative symptoms to check next")
        st.caption("Symptoms not reported yet, ranked by how much they would sharpen the top answer.")
        st.dataframe(
            pd.DataFrame(
                {
                    "symptom": [c.symptom for c in result.follow_up_symptoms],
                    "would change top probability by": [
                        f"{c.contribution:+.1%}" for c in result.follow_up_symptoms
                    ],
                }
            ),
            hide_index=True,
            use_container_width=True,
        )


def main() -> None:
    """Compose the page."""
    st.title("🩺 Disease Predictor")
    st.caption("Symptom-to-diagnosis ranking with per-symptom explanations.")
    st.error(DISCLAIMER, icon="⚠️")

    try:
        predictor = load_predictor()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.code("dp train --synthetic   # train a demo model in a few seconds", language="bash")
        st.stop()
        return

    render_sidebar(predictor)

    selected = st.multiselect(
        "Select the symptoms you are experiencing",
        options=predictor.symptoms,
        help="Type to search. Three or more symptoms give a meaningful ranking.",
    )
    left, right = st.columns([1, 3])
    top_k = left.slider("Shortlist length", min_value=1, max_value=10, value=3)
    right.write("")

    if st.button("Predict", type="primary", disabled=not selected):
        result = predictor.predict(selected, top_k=top_k, explain=True, follow_up=True)
        render_predictions(result)
    elif not selected:
        st.info("Select at least one symptom to get a prediction.")


if __name__ == "__main__":
    main()
