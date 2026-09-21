from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

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


def train_model():
    df = pd.read_csv(DATA_PATH)

    if "salary_in_usd" not in df.columns:
        raise ValueError("The dataset does not contain the 'salary_in_usd' target column.")

    X = df[FEATURE_COLUMNS]
    y = df["salary_in_usd"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    categorical_features = X.select_dtypes(include=["object", "string"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ],
        remainder="passthrough",
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", Ridge(alpha=1.0)),
        ]
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, predictions)

    print("Model training complete.")
    print(f"MAE: ${mae:,.0f}")
    print(f"MSE: ${mse:,.0f}")
    print(f"RMSE: ${rmse:,.0f}")
    print(f"R2 Score: {r2:.4f}")

    joblib.dump(model, MODEL_PATH)
    print(f"Saved model to: {MODEL_PATH}")
    return model


if __name__ == "__main__":
    train_model()
