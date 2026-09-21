from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
DATA_PATH = APP_DIR / "jobs_in_data.csv"
MODEL_PATH = APP_DIR / "salary_model.joblib"

FEATURE_COLUMNS = [
    "work_year",
    "job_title",
    "job_category",
    "employee_residence",
    "experience_level",
    "employment_type",
    "work_setting",
    "company_location",
    "company_size",
]


def train_if_missing():
    if not MODEL_PATH.exists():
        from train_salary_model import train_model

        train_model()


@st.cache_resource
def load_model_and_data():
    train_if_missing()
    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(DATA_PATH)
    return model, df


st.set_page_config(page_title="Salary Predictor", page_icon="💰", layout="wide")
st.title("Data Salary Predictor")
st.caption("Estimate a data professional's annual salary in USD from career and company attributes.")

model, df = load_model_and_data()

job_titles = sorted(df["job_title"].dropna().unique().tolist())
job_categories = sorted(df["job_category"].dropna().unique().tolist())
residences = sorted(df["employee_residence"].dropna().unique().tolist())
experience_levels = sorted(df["experience_level"].dropna().unique().tolist())
employment_types = sorted(df["employment_type"].dropna().unique().tolist())
work_settings = sorted(df["work_setting"].dropna().unique().tolist())
company_locations = sorted(df["company_location"].dropna().unique().tolist())
company_sizes = sorted(df["company_size"].dropna().unique().tolist())

with st.form("salary_form"):
    col1, col2 = st.columns(2)

    with col1:
        work_year = st.number_input("Work year", min_value=2020, max_value=2025, value=2024)
        job_title = st.selectbox("Job title", job_titles)
        job_category = st.selectbox("Job category", job_categories)
        employee_residence = st.selectbox("Employee residence", residences)
        experience_level = st.selectbox("Experience level", experience_levels)

    with col2:
        employment_type = st.selectbox("Employment type", employment_types)
        work_setting = st.selectbox("Work setting", work_settings)
        company_location = st.selectbox("Company location", company_locations)
        company_size = st.selectbox("Company size", company_sizes)

    submitted = st.form_submit_button("Predict salary", type="primary")

if submitted:
    payload = pd.DataFrame(
        [
            {
                "work_year": work_year,
                "job_title": job_title,
                "job_category": job_category,
                "employee_residence": employee_residence,
                "experience_level": experience_level,
                "employment_type": employment_type,
                "work_setting": work_setting,
                "company_location": company_location,
                "company_size": company_size,
            }
        ]
    )

    prediction = float(model.predict(payload)[0])
    st.success(f"Estimated salary: ${prediction:,.0f} USD per year")

    st.caption(
        "This estimate is based on a Ridge regression model trained on the salary dataset in this project."
    )
