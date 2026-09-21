from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


APP_DIR = Path(__file__).resolve().parent

st.set_page_config(page_title="Student Dropout Risk", page_icon=":bar_chart:", layout="wide")


@st.cache_resource
def load_artifacts():
    return {
        "model": joblib.load(APP_DIR / "dropout_model.pkl"),
        "scaler": joblib.load(APP_DIR / "scaler.pkl"),
        "feature_columns": joblib.load(APP_DIR / "feature_columns.pkl"),
        "categorical_cols": joblib.load(APP_DIR / "categorical_cols.pkl"),
        "numeric_cols": joblib.load(APP_DIR / "numeric_cols.pkl"),
        "categorical_defaults": joblib.load(APP_DIR / "categorical_defaults.pkl"),
        "numeric_defaults": joblib.load(APP_DIR / "numeric_defaults.pkl"),
        "threshold": float(joblib.load(APP_DIR / "decision_threshold.pkl")),
    }


COURSE_CODES = {
    "Biotechnology": 171,
    "Computer Science": 9254,
    "Social Work": 9070,
    "Agricultural Science": 9773,
    "Mass Communication": 8014,
    "Veterinary Medicine": 9991,
    "Computer Engineering": 9500,
    "Animal Science": 9238,
    "Business Administration": 9670,
    "Hospitality Management": 9085,
    "Nursing Science": 9130,
    "Dental Technology": 9556,
    "Marketing": 9147,
    "Primary Education": 33,
}

FIELD_LABELS = {
    "Age at enrollment": "Age at enrollment",
    "Admission grade": "Admission grade",
    "Previous qualification (grade)": "Previous qualification grade",
    "Curricular units 1st sem (grade)": "First-semester grade",
    "Curricular units 2nd sem (grade)": "Second-semester grade",
    "Curricular units 1st sem (enrolled)": "First-semester units enrolled",
    "Curricular units 1st sem (evaluations)": "First-semester evaluations",
    "Curricular units 1st sem (approved)": "First-semester units approved",
    "Curricular units 2nd sem (enrolled)": "Second-semester units enrolled",
    "Curricular units 2nd sem (evaluations)": "Second-semester evaluations",
    "Curricular units 2nd sem (approved)": "Second-semester units approved",
}


def category_options(field, feature_columns, default):
    prefix = f"{field}_"
    options = [column[len(prefix):] for column in feature_columns if column.startswith(prefix)]
    return sorted(set(options + [default]))


def number_input_for(field, default):
    label = FIELD_LABELS.get(field, field)
    if field in {"Age at enrollment", "Application mode", "Application order"}:
        return st.number_input(label, value=int(default), step=1, min_value=0)
    if "grade" in field.lower() or "grade" in label.lower():
        return st.number_input(label, value=float(default), min_value=0.0, max_value=200.0, step=0.1)
    if "units" in field or "evaluations" in field or "occupation" in field:
        return st.number_input(label, value=float(default), min_value=0.0, step=1.0)
    return st.number_input(label, value=float(default), step=0.01, format="%.3f")


def build_input_row(values, artifacts):
    categorical = artifacts["categorical_cols"]
    numeric = artifacts["numeric_cols"]
    feature_columns = artifacts["feature_columns"]

    row = {field: values[field] for field in categorical}
    row.update({field: values[field] for field in numeric})
    row["Course"] = float(COURSE_CODES.get(values["Course_Nigerian"], artifacts["numeric_defaults"]["Course"]))

    row["Grade trend"] = row["Curricular units 2nd sem (grade)"] - row["Curricular units 1st sem (grade)"]
    row["1st sem approval ratio"] = (
        row["Curricular units 1st sem (approved)"] / row["Curricular units 1st sem (enrolled)"]
        if row["Curricular units 1st sem (enrolled)"] > 0 else 0
    )
    row["2nd sem approval ratio"] = (
        row["Curricular units 2nd sem (approved)"] / row["Curricular units 2nd sem (enrolled)"]
        if row["Curricular units 2nd sem (enrolled)"] > 0 else 0
    )
    row["Financial risk score"] = sum([
        row["Debtor"] == "Yes",
        row["Tuition fees up to date"] == "No",
        row["Scholarship holder"] == "No",
    ])

    frame = pd.DataFrame([row])
    encoded = pd.get_dummies(frame, columns=categorical, drop_first=True)
    encoded = encoded.reindex(columns=feature_columns, fill_value=0)
    numeric_present = [field for field in numeric if field in encoded.columns]
    encoded[numeric_present] = artifacts["scaler"].transform(encoded[numeric_present])
    return encoded


st.markdown(
    """
    <style>
    .block-container {max-width: 1120px; padding-top: 2.5rem;}
    [data-testid="stMetricValue"] {font-size: 2rem;}
    .risk-note {padding: 1rem 1.2rem; border-radius: 0.5rem; background: #f4f6f8;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Student Dropout Risk Predictor")
st.caption("An early-warning estimate for Nigerian higher-education settings")
st.write("Enter the student's available information. Fields not supplied in the form use the training-data median or most common value.")

try:
    artifacts = load_artifacts()
except FileNotFoundError as error:
    st.error(f"A required model artifact is missing: {error.filename}")
    st.stop()

with st.form("prediction_form"):
    st.subheader("Student profile")
    profile_cols = st.columns(3)
    values = {}
    profile_fields = [
        "Course_Nigerian", "Gender", "Age at enrollment", "Marital status",
        "Nationality", "Daytime/evening attendance", "Previous qualification",
        "Mother's qualification", "Father's qualification", "Displaced",
        "Educational special needs", "International",
    ]
    for index, field in enumerate(profile_fields):
        with profile_cols[index % 3]:
            if field in artifacts["categorical_cols"]:
                default = artifacts["categorical_defaults"][field]
                options = category_options(field, artifacts["feature_columns"], default)
                values[field] = st.selectbox(FIELD_LABELS.get(field, field), options, index=options.index(default))
            else:
                values[field] = number_input_for(field, artifacts["numeric_defaults"][field])

    st.subheader("Academic and financial indicators")
    academic_cols = st.columns(3)
    academic_fields = [
        "Admission grade", "Previous qualification (grade)",
        "Curricular units 1st sem (grade)", "Curricular units 2nd sem (grade)",
        "Curricular units 1st sem (enrolled)", "Curricular units 1st sem (evaluations)",
        "Curricular units 1st sem (approved)", "Curricular units 2nd sem (enrolled)",
        "Curricular units 2nd sem (evaluations)", "Curricular units 2nd sem (approved)",
    ]
    for index, field in enumerate(academic_fields):
        with academic_cols[index % 3]:
            values[field] = number_input_for(field, artifacts["numeric_defaults"][field])

    finance_cols = st.columns(3)
    for index, field in enumerate(["Debtor", "Tuition fees up to date", "Scholarship holder"]):
        with finance_cols[index]:
            default = artifacts["categorical_defaults"][field]
            values[field] = st.selectbox(field, ["Yes", "No"], index=["Yes", "No"].index(default))

    with st.expander("Additional model inputs"):
        extra_cols = st.columns(3)
        for index, field in enumerate(artifacts["numeric_cols"]):
            if field in values or field in {"Course", "Grade trend", "1st sem approval ratio", "2nd sem approval ratio", "Financial risk score"}:
                continue
            with extra_cols[index % 3]:
                values[field] = number_input_for(field, artifacts["numeric_defaults"][field])

    submitted = st.form_submit_button("Assess dropout risk", type="primary", use_container_width=True)

if submitted:
    input_frame = build_input_row(values, artifacts)
    probability = float(artifacts["model"].predict_proba(input_frame)[0, 1])
    is_at_risk = probability >= artifacts["threshold"]

    st.divider()
    result_cols = st.columns([1, 1, 1])
    with result_cols[0]:
        st.metric("Estimated dropout risk", f"{probability:.1%}")
    with result_cols[1]:
        st.metric("Model threshold", f"{artifacts['threshold']:.1%}")
    with result_cols[2]:
        st.metric("Assessment", "At risk" if is_at_risk else "Lower risk")

    if is_at_risk:
        st.warning("This student is above the model's early-warning threshold. Consider a timely academic or financial support check-in.")
    else:
        st.success("This student is below the model's early-warning threshold. Continue normal academic monitoring.")

    st.caption("This is a screening estimate, not a definitive outcome. Use it alongside advisor judgement and direct student support.")