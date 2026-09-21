

Dropout predictor readme · MD
Student Dropout Risk Predictor — Nigerian Context
An early-warning system that predicts a student's dropout risk from academic, financial, and demographic data — reframed from a real dataset into a Nigerian higher-education context.

Problem
Schools and academic advisors often only learn a student is struggling after it's too late to intervene. This project builds a model that flags at-risk students early, using signals already available to a school (grades, unit completion, attendance type, fee/scholarship status) — so a counsellor can act before dropout happens, not after.

Dataset
Built on the UCI "Predict Students' Dropout and Academic Success" dataset (Realinho et al., 2021) — 4,424 real student records, 37 features, zero missing values. No labeled Nigerian dropout dataset exists publicly (Nigeria's MICS survey data requires manual UNICEF approval), so this dataset was reframed instead:

Categorical codes (course, prior qualification, parents' qualifications) decoded into Nigerian-equivalent terms (e.g. WAEC/NECO instead of the original Portuguese secondary certificates)
Grades rescaled to a Nigerian percentage scale
Original Portuguese macroeconomic columns replaced with real, verified Nigerian GDP growth, inflation, and unemployment figures
This limitation — a non-Nigerian dataset, reframed rather than natively Nigerian — is stated explicitly rather than hidden, since the alternative (a fully synthetic or unverified dataset) would be worse for a real-world tool.

Approach
EDA: target class balance, dropout rate by course/category, grade trends by outcome, correlation analysis, dropout rate vs. Nigerian GDP growth by year
Feature engineering: grade trend (2nd sem − 1st sem), per-semester approval ratios, a composite financial risk score (debtor status + tuition arrears + no scholarship)
Models compared: Logistic Regression, Random Forest, XGBoost (baseline), a soft-voting ensemble, and a final tuned model
Final model: XGBoost + SMOTE (oversampling applied to the training set only, never the test set) + RandomizedSearchCV hyperparameter tuning + decision-threshold tuning (optimized for F1 on the Dropout class, not the default 0.5 cutoff)
Explainability: SHAP (TreeExplainer) — both a global feature-importance view across the test set, and a per-student waterfall breakdown showing exactly why an individual prediction was made
Key Decision
Optimized for Recall on the Dropout class over raw accuracy throughout. Reasoning: for an early-warning system, a false negative (missing a student who actually drops out) is a costlier mistake than a false positive (flagging a student who turns out fine) — so the decision threshold was deliberately tuned below 0.5 to trade some precision for meaningfully higher recall.

Results
Model	Accuracy	F1 (Dropout)	Recall (Dropout)
Logistic Regression	—	—	higher recall, more false alarms
Random Forest	—	—	higher precision, misses more true dropouts
XGBoost (baseline)	—	—	best balance of the baselines
Tuned XGBoost + SMOTE + threshold tuning (final)	89.7%	0.844	87.0%
Try It Yourself
The notebook includes a predict_dropout_risk() function — pass in a student's course, grades, attendance type, and financial status, and it returns a risk prediction using the same final model, with SHAP explaining exactly which factors drove that specific prediction.

Tech Stack
Python, Pandas, NumPy, scikit-learn, XGBoost, imbalanced-learn (SMOTE), SHAP, Matplotlib, Seaborn

Live Demo
[Will be Added once deployed]


