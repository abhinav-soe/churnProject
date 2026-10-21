# Customer Churn Prediction Dashboard

A machine-learning project for analyzing customer churn and identifying customers who may be at risk of leaving a service.

The project combines data cleaning, feature engineering, model comparison, and an interactive Streamlit dashboard. It supports synthetic Telco-style data for experimentation and a real Telco Customer Churn dataset supplied by the user.

## Features

- **Data preparation:** Clean missing values, blank fields, and duplicate records in the input data.
- **Feature engineering:** Prepare customer attributes for model training and prediction through shared preprocessing logic.
- **Model comparison:** Train and compare three machine-learning models alongside a baseline.
- **Model evaluation:** Compare cross-validation ROC-AUC and evaluate precision, recall, F1-score, and ROC-AUC.
- **Interactive dashboard:** Explore the data, inspect model performance, and make predictions for individual customers.
- **At-risk customer list:** Review customers flagged by the trained model as potentially likely to churn.
- **Reusable artifacts:** Save the best-performing model and related artifacts for use by the dashboard.

## How It Works

The project follows a data-to-prediction workflow.

1. **Load the data:** Generate synthetic Telco-style customer records or provide a real dataset at `data/churn.csv`.
2. **Clean and prepare features:** `features.py` handles data cleaning and feature engineering shared by the training script and dashboard.
3. **Train candidate models:** `train.py` trains three models and a baseline using the prepared dataset.
4. **Evaluate performance:** Candidate models are compared using cross-validation ROC-AUC and classification metrics, including precision, recall, F1-score, and ROC-AUC.
5. **Save the best model:** The training pipeline saves the selected model and artifacts to `models/`.
6. **Explore predictions:** `app.py` provides a Streamlit interface for exploratory data analysis, model-performance review, individual customer predictions, and an at-risk customer list.

## Tech Stack

- **Language:** Python
- **Machine learning:** scikit-learn and the libraries listed in `requirements.txt`
- **Data processing:** pandas and NumPy, if included in the project dependencies
- **Dashboard:** Streamlit
- **Evaluation:** Cross-validation ROC-AUC, precision, recall, F1-score, and ROC-AUC

## Getting Started

### Prerequisites

- Python installed in a version compatible with the project dependencies
- pip
- The project source code
- Optional: the Kaggle Telco Customer Churn dataset

### 1. Clone the repository

```bash id="k9w2je"
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd <YOUR_REPOSITORY_DIRECTORY>
```

Replace the placeholders with your repository URL and local folder name.

### 2. Install dependencies

```bash id="f1p4sl"
python -m pip install -r requirements.txt
```

### 3. Generate sample data

To create synthetic Telco-style data for experimentation, run:

```bash id="m0drgf"
python generate_data.py
```

The generated data includes missing or blank values and duplicate records so the cleaning pipeline can be exercised.

Alternatively, provide the real dataset as described below.

### 4. Train and compare models

```bash id="p2n31a"
python train.py
```

This runs the training and evaluation workflow, compares the candidate models with the baseline, and saves the selected model and artifacts to the `models/` directory.

Run training again whenever you change the dataset or relevant preprocessing and model settings.

### 5. Launch the dashboard

```bash id="n5wd2a"
streamlit run app.py
```

Open the local URL displayed by Streamlit in your terminal.

The dashboard provides views for exploratory data analysis, model performance, single-customer predictions, and customers identified as potentially at risk of churn.

## Using the Real Telco Customer Churn Dataset

You can use the Telco Customer Churn dataset available through Kaggle.

1. Download the dataset from Kaggle.
2. Place the CSV file at `data/churn.csv`.
3. Check that its column names and data types match the expectations in `features.py`.
4. If necessary, update the cleaning and feature-engineering logic to match the dataset's schema.
5. Run the training script again:

```bash id="cz1c2u"
python train.py
```

6. Launch the dashboard:

```bash id="l2l0fj"
streamlit run app.py
```

The project may require adjustments to column names or preprocessing logic if the downloaded CSV differs from the expected schema.

## Model Evaluation

The training pipeline compares candidate models using the metrics below.

| Metric | Purpose |
|---|---|
| Cross-validation ROC-AUC | Compare how well candidate models distinguish between churn classes across validation folds |
| Precision | Measure the proportion of customers predicted to churn who actually churn |
| Recall | Measure the proportion of actual churners identified by the model |
| F1-score | Balance precision and recall |
| ROC-AUC | Measure the model's ability to distinguish between the two classes across classification thresholds |

In churn prediction, precision and recall represent different business trade-offs. Higher recall may help identify more at-risk customers, while higher precision can reduce the number of customers incorrectly flagged for intervention.

Actual scores should be reported from the results produced by your own training run.

## Project Structure

| File or directory | Purpose |
|---|---|
| `generate_data.py` | Generate synthetic Telco-style data containing data-quality issues |
| `features.py` | Clean data and perform feature engineering shared by training and inference |
| `train.py` | Train and compare candidate models, evaluate performance, and save artifacts |
| `app.py` | Streamlit dashboard for EDA, model performance, customer prediction, and at-risk customers |
| `data/churn.csv` | Optional input location for the real customer churn dataset |
| `models/` | Saved model and training artifacts |
| `requirements.txt` | Python dependencies |

## Limitations and Considerations

- **Synthetic versus real data:** Results from synthetic data may not reflect performance on real customer populations.
- **Dataset compatibility:** The real dataset may require column-name, type, and preprocessing adjustments.
- **Class imbalance:** Churn datasets may contain fewer churners than non-churners, so accuracy alone may be misleading.
- **Prediction versus certainty:** A predicted churn risk is not proof that a customer will leave.
- **Business context:** Model thresholds should be selected with intervention costs and the consequences of false positives and false negatives in mind.

## Potential Improvements

- Add a confusion matrix and precision-recall curve to the dashboard.
- Tune classification thresholds based on the cost of missed churners and unnecessary interventions.
- Add model explainability with feature importance or SHAP.
- Track data drift and monitor performance as new customer data becomes available.
- Add automated tests for preprocessing, training, and prediction.
- Record experiment settings and model metrics to improve reproducibility.

## License and Dataset Attribution

Add the appropriate license for this repository. If you use a third-party dataset, follow its licensing terms and attribution requirements. Do not commit customer data or other sensitive information to the repository.
