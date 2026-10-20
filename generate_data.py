"""Generate a synthetic Telco-style churn dataset (swap in the Kaggle
'Telco Customer Churn' CSV at data/churn.csv if you prefer real data)."""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 7000

def yn(p, n=N):
    return np.where(rng.random(n) < p, "Yes", "No")

df = pd.DataFrame({
    "customerID": [f"C{100000 + i}" for i in range(N)],
    "gender": rng.choice(["Male", "Female"], N),
    "SeniorCitizen": (rng.random(N) < 0.16).astype(int),
    "Partner": yn(0.48),
    "Dependents": yn(0.30),
    "tenure": np.clip(rng.gamma(2.0, 18, N).astype(int), 0, 72),
    "PhoneService": yn(0.90),
    "InternetService": rng.choice(["DSL", "Fiber optic", "No"], N, p=[0.34, 0.44, 0.22]),
    "Contract": rng.choice(["Month-to-month", "One year", "Two year"], N, p=[0.55, 0.21, 0.24]),
    "PaperlessBilling": yn(0.59),
    "PaymentMethod": rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        N, p=[0.34, 0.23, 0.22, 0.21]),
})
df["MultipleLines"] = np.where(df.PhoneService == "No", "No phone service", yn(0.42))
ADDONS = [("OnlineSecurity", .35), ("OnlineBackup", .40), ("DeviceProtection", .40),
          ("TechSupport", .35), ("StreamingTV", .45), ("StreamingMovies", .45)]
for col, p in ADDONS:
    df[col] = np.where(df.InternetService == "No", "No internet service", yn(p))

base = 20 + 25 * (df.InternetService == "DSL") + 55 * (df.InternetService == "Fiber optic")
extras = sum((df[c] == "Yes") * 5 for c, _ in ADDONS)
df["MonthlyCharges"] = (base + extras + rng.normal(0, 4, N)).round(2).clip(18, 120)
df["TotalCharges"] = (df.MonthlyCharges * df.tenure * rng.uniform(0.95, 1.05, N)).round(2)

# churn probability from realistic drivers
z = (-2.0
     + 1.5 * (df.Contract == "Month-to-month") - 1.2 * (df.Contract == "Two year")
     + 0.8 * (df.InternetService == "Fiber optic")
     + 0.6 * (df.PaymentMethod == "Electronic check")
     + 0.5 * (df.TechSupport == "No") + 0.4 * (df.OnlineSecurity == "No")
     + 0.3 * df.SeniorCitizen + 0.3 * (df.PaperlessBilling == "Yes")
     - 0.035 * df.tenure + 0.008 * (df.MonthlyCharges - 65))
df["Churn"] = np.where(rng.random(N) < 1 / (1 + np.exp(-z)), "Yes", "No")

# inject realistic mess for the cleaning step
df["TotalCharges"] = df["TotalCharges"].astype(object)
df.loc[df.tenure == 0, "TotalCharges"] = " "           # blank strings, like the real Telco data
df.loc[rng.choice(N, 40, replace=False), "MonthlyCharges"] = np.nan
df = pd.concat([df, df.sample(25, random_state=1)])    # duplicate rows

df.to_csv("data/churn.csv", index=False)
print(df.shape, "| churn rate:", (df.Churn == "Yes").mean().round(3))
