# Customer Segmentation with K-Means

A small machine-learning application that groups customers into behavioral segments using the K-Means clustering algorithm. The project includes a Jupyter notebook for data generation, exploration, model training, evaluation, and serialization, plus a Streamlit app for making predictions for an individual customer.

## Project Overview

Customer segmentation can help a business tailor communication, promotions, and services to different types of customers. This project uses four customer attributes to assign a new customer to one of three clusters:

- Age
- Average spend
- Visits per week
- Promotion interest

The trained model is saved as `kmeans.pkl` and loaded by the Streamlit application.

## Project Structure

```text
Customer-Segmentation/
├── app.py                         # Streamlit prediction application
├── Customer segmentation Kmeans.ipynb  # Data generation, training, and evaluation
├── kmeans.pkl                     # Serialized trained K-Means model
├── requirements.txt               # Python dependencies
└── README.md                      # Project documentation
```

## How It Works

1. The notebook generates a sample dataset containing 100 customers.
2. The four behavioral features are selected for clustering.
3. K-Means models are evaluated for different cluster counts using the elbow method.
4. A three-cluster K-Means model is trained with `random_state=42`.
5. The model is evaluated with a silhouette score.
6. The trained model is saved to `kmeans.pkl`.
7. `app.py` loads the model and predicts a cluster from values entered in the web form.

This is an unsupervised-learning project: the clusters are discovered from the data rather than learned from pre-existing customer labels.

## Installation

Python 3.9 or newer is recommended.

From this directory, create and activate a virtual environment:

```bash
cd Customer-Segmentation
python -m venv .venv
source .venv/bin/activate       # Linux/macOS
# .venv\Scripts\activate      # Windows PowerShell
```

Install the dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run the Streamlit App

Run the command from the `Customer-Segmentation` directory so that the app can find `kmeans.pkl`:

```bash
streamlit run app.py
```

Streamlit will display a local URL, usually `http://localhost:8501`. Enter the customer details and select **Predict Cluster** to see the prediction.

### Input Ranges

| Input | Allowed range | Description |
|---|---:|---|
| Age | 18 to 100 | Customer age in years |
| Average Spend | 0 to 1,000 | Average customer spend |
| Visits per Week | 0 to 20 | Number of weekly visits |
| Promotion Interest | 1 to 10 | Interest in promotions |

## Run the Notebook

To reproduce the exploratory analysis and training workflow:

```bash
jupyter notebook "Customer segmentation Kmeans.ipynb"
```

Run the cells in order. The notebook creates the sample data, trains the model, calculates the silhouette score, and writes `kmeans.pkl` to the project directory.

## Cluster Labels

K-Means cluster numbers do not have an inherent business meaning. The notebook currently assigns names using this mapping:

```text
0 -> Daily
1 -> Promotion
2 -> Weekend
```

The current Streamlit app uses a different mapping for the second and third clusters (`1 -> Weekend`, otherwise `Promotion`). These mappings should be made consistent before using the predictions for business decisions. The labels are descriptive names only and should be validated against real customer behavior.

## Limitations

- The training data is synthetic, so the clusters may not represent real customer behavior.
- The model is not currently retrained from data uploaded by the user.
- The model is saved with Python pickle; load model files only from trusted sources.
- No feature scaling is applied, so features with larger numeric ranges can influence the clustering more strongly.
- Cluster names are manually assigned after training and are not automatically learned by K-Means.

## Technologies

- Python
- NumPy and pandas
- scikit-learn
- Matplotlib
- Streamlit
- Jupyter Notebook
