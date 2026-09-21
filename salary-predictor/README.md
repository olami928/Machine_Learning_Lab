# Salary Predictor

This project predicts a data professional's annual salary in USD using a regression model trained on the `jobs_in_data.csv` dataset.

## What the project does

- Loads the salary dataset
- Prepares the input features
- Encodes categorical values with a scikit-learn preprocessing pipeline
- Trains a Ridge regression model
- Saves the trained model as `salary_model.joblib`
- Exposes a Streamlit app for live salary prediction

## Project structure

```text
salary-predictor/
├── app.py                 # Streamlit web app for salary prediction
├── jobs_in_data.csv       # Dataset used for training
├── README.md              # Project documentation
├── requirements.txt       # Python dependencies
├── salary_model.joblib    # Generated trained model
└── train_salary_model.py  # Training script for the model
```

## Error that was fixed

The original notebook workflow had a broken training sequence: it dropped target columns and then reused variables that were never created in a consistent pipeline. The result was `NameError` and `KeyError` exceptions during model training.

The corrected approach is:

1. Load the CSV safely with pandas
2. Select only valid feature columns
3. Set `salary_in_usd` as the target
4. Encode text columns with `OneHotEncoder`
5. Train the model in a single pipeline
6. Save the pipeline and reuse it in the app

## Installation

From the project folder:

```bash
cd salary-predictor
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the model trainer

```bash
python train_salary_model.py
```

## Run the Streamlit app

```bash
streamlit run app.py
```

The app will open in your browser and let you choose a job profile to estimate salary.

## Example features used

- Work year
- Job title
- Job category
- Employee residence
- Experience level
- Employment type
- Work setting
- Company location
- Company size

## Model details

- Model type: Ridge Regression
- Target variable: `salary_in_usd`
- Encoding: OneHotEncoder with `handle_unknown='ignore'`
- Evaluation metric: R² on a test split

## Dependencies

- Python 3.10+
- pandas
- scikit-learn
- streamlit
- joblib
