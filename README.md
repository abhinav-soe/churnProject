# Customer Churn Prediction Dashboard

## Run
```bash
pip install -r requirements.txt
python generate_data.py     # or drop the Kaggle Telco CSV at data/churn.csv
python train.py             # trains 3 models + baseline, saves best to models/
streamlit run app.py
```

## Structure
| File | Purpose |
|---|---|
| `generate_data.py` | Synthetic Telco-style data (with blanks, NaNs, duplicates to clean) |
| `features.py` | Cleaning + feature engineering, shared by training and dashboard |
| `train.py` | Model comparison (CV ROC-AUC, precision/recall/F1/ROC-AUC), saves artifacts |
| `app.py` | Streamlit dashboard: EDA, model performance, single-customer predictor, at-risk list |

## Using the real dataset
Download "Telco Customer Churn" from Kaggle, save as `data/churn.csv`, rerun `train.py`.
You may need to adjust column names in `features.py` if they differ.
