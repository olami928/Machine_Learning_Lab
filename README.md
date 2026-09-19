# ML Portfolio — Alabi Oladimeji Taofeek

A growing collection of AI/ML projects spanning tabular classification, computer vision, and NLP — from problem framing through deployment.

**Background:** Wood & Biomaterials Engineering (B.Sc., University of Ibadan) working at the intersection of domain engineering and machine learning. ALX Software Engineering certificate, Applied AI certificate.

[LinkedIn](https://bit.ly/3RRYHHi) · [GitHub](https://github.com/olami928) · Oladimejialabi45@gmail.com

---

## Projects

| Project | Problem | Key Result | Stack |
|---|---|---|---|
| [Student Dropout Risk Predictor](./dropout-predictor) | Early-warning system flagging at-risk students from academic/financial data | 89.7% accuracy, 0.844 F1, 87.0% recall on Dropout class | XGBoost, SMOTE, SHAP |
| [Timber Species & End-Use Identification](./timber-thesis) | Multi-output image classification identifying Nigerian timber species and probable end-use | In progress — multi-task CNN, uncertainty-weighted loss | PyTorch, transfer learning (ShuffleNetV2, ResNet18) |
| [Sentiment Classifier](./sentiment-classifier) | Fine-tuned transformer for movie review sentiment | Fine-tuned DistilBERT on IMDB reviews | Hugging Face Transformers |

*New projects are added here as their own row, following the same structure below.*

---

## Structure

Every project lives in its own top-level folder, named after the project, and is self-contained:

```
<project-name>/
├── README.md          # Problem, approach, results, key decisions
├── train.py            # or notebook — the modeling/training code
├── app.py                # if deployed
└── requirements.txt
```

The root `README.md` (this file) stays a high-level index — the projects table above, kept current as new work is added. Depth always lives inside each project's own README, not here.

## Skills

Reflects tools actually used across the projects above — grows as the portfolio does.

**Languages & Core:** Python, SQL
**ML/DL:** scikit-learn, XGBoost, PyTorch, TensorFlow/Keras, Transfer Learning (CNNs), Hugging Face Transformers
**Data:** Pandas, NumPy, SMOTE, SHAP
**Deployment:** Streamlit, Flask, Git/GitHub
**Tools:** Jupyter/Colab, Linux, APIs

## How to Explore

Each project's README covers problem, approach, results, and the reasoning behind key decisions — start with whichever matches what you're evaluating for. The Dropout Predictor is currently the most complete end-to-end example (EDA → modeling → tuning → explainability).

---

*Open to AI/ML roles. Reach out via [LinkedIn](https://bit.ly/3RRYHHi) or email above.*
