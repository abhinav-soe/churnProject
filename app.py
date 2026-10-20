import json
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (confusion_matrix, f1_score, precision_score,
                             recall_score, roc_curve)

from features import FEATURES, clean, engineer

st.set_page_config(page_title="Churn Prediction Dashboard", page_icon="📉", layout="wide")
sns.set_theme(style="whitegrid")


@st.cache_data
def load():
    raw = pd.read_csv("data/churn.csv")
    df = engineer(clean(raw))
    return raw, df, pd.read_csv("models/test_scored.csv"), pd.read_csv("models/test_predictions.csv"), \
        pd.read_csv("models/feature_importance.csv"), json.load(open("models/metrics.json"))


raw, df, scored, preds, importance, metrics = load()
model = joblib.load("models/best_model.joblib")
best = metrics["best_model"]

st.title("📉 Customer Churn Prediction Dashboard")
st.caption(f"Best model: **{best}** (selected on test ROC-AUC)")

# ---- sidebar filters (apply to the EDA tab) ----
st.sidebar.header("Filters")
contracts = st.sidebar.multiselect("Contract", df.Contract.unique(), default=list(df.Contract.unique()))
internet = st.sidebar.multiselect("Internet service", df.InternetService.unique(),
                                  default=list(df.InternetService.unique()))
view = df[df.Contract.isin(contracts) & df.InternetService.isin(internet)]

tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview & EDA", "🤖 Model performance",
                                  "🔮 Predict a customer", "🚨 At-risk customers"])

# ---------------- TAB 1: EDA ----------------
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers", f"{len(view):,}")
    c2.metric("Churn rate", f"{view.Churn.mean():.1%}")
    c3.metric("Avg monthly charge", f"${view.MonthlyCharges.mean():.2f}")
    c4.metric("Avg tenure (months)", f"{view.tenure.mean():.1f}")

    a, b = st.columns(2)
    with a:
        st.subheader("Churn rate by contract")
        fig, ax = plt.subplots(figsize=(5, 3.2))
        sns.barplot(data=view, x="Contract", y="Churn", errorbar=None, ax=ax, color="#d9534f")
        ax.set_ylabel("Churn rate"); ax.set_xlabel("")
        st.pyplot(fig)
    with b:
        st.subheader("Churn rate by tenure group")
        fig, ax = plt.subplots(figsize=(5, 3.2))
        order = ["0-1y", "1-2y", "2-4y", "4y+"]
        sns.barplot(data=view, x="tenure_group", y="Churn", order=order, errorbar=None, ax=ax, color="#f0ad4e")
        ax.set_ylabel("Churn rate"); ax.set_xlabel("")
        st.pyplot(fig)

    a, b = st.columns(2)
    with a:
        st.subheader("Monthly charges: churned vs retained")
        fig, ax = plt.subplots(figsize=(5, 3.2))
        sns.kdeplot(data=view, x="MonthlyCharges", hue=view.Churn.map({0: "Retained", 1: "Churned"}),
                    fill=True, common_norm=False, ax=ax)
        st.pyplot(fig)
    with b:
        st.subheader("Churn rate by payment method")
        fig, ax = plt.subplots(figsize=(5, 3.2))
        sns.barplot(data=view, y="PaymentMethod", x="Churn", errorbar=None, ax=ax, color="#5bc0de")
        ax.set_xlabel("Churn rate"); ax.set_ylabel("")
        st.pyplot(fig)

    with st.expander("Data cleaning summary"):
        st.write(f"- Raw rows: **{len(raw):,}** → after cleaning: **{len(df):,}** "
                 f"(duplicates removed: {len(raw) - len(df)})")
        st.write(f"- Blank `TotalCharges` strings (new customers): **{(raw.TotalCharges.astype(str).str.strip() == '').sum()}**")
        st.write(f"- Missing `MonthlyCharges` imputed with median: **{raw.MonthlyCharges.isna().sum()}**")
        st.write("- Engineered features: `avg_monthly_spend`, `num_services`, `charge_per_service`, "
                 "`tenure_group`, `is_autopay`, `new_month_to_month`; `num_services` now counts all 6 internet add-ons")

# ---------------- TAB 2: MODEL PERFORMANCE ----------------
with tab2:
    st.subheader("Model comparison (hold-out test set, threshold 0.5)")
    res = pd.DataFrame(metrics["results"]).T[["accuracy", "precision", "recall", "f1", "roc_auc", "cv_roc_auc"]]
    st.dataframe(res.style.format("{:.3f}").highlight_max(axis=0, color="#cdeccd"), width="stretch")

    a, b = st.columns(2)
    with a:
        st.subheader("ROC curves")
        fig, ax = plt.subplots(figsize=(5, 4))
        for name in metrics["results"]:
            if name.startswith("Baseline"):
                continue
            fpr, tpr, _ = roc_curve(preds.y_true, preds[name])
            ax.plot(fpr, tpr, label=f"{name} ({metrics['results'][name]['roc_auc']:.3f})")
        ax.plot([0, 1], [0, 1], "k--", alpha=.4)
        ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate"); ax.legend(fontsize=8)
        st.pyplot(fig)
    with b:
        st.subheader("What drives churn? (permutation importance)")
        top = importance.head(10)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.barplot(data=top, y="feature", x="importance", ax=ax, color="#4c72b0")
        ax.set_xlabel("Drop in ROC-AUC when shuffled"); ax.set_ylabel("")
        st.pyplot(fig)

    st.subheader(f"Tune the decision threshold ({best})")
    thr = st.slider("Flag a customer as 'will churn' when probability ≥", 0.05, 0.95, 0.50, 0.05)
    pred = (preds[best] >= thr).astype(int)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Precision", f"{precision_score(preds.y_true, pred, zero_division=0):.3f}")
    m2.metric("Recall", f"{recall_score(preds.y_true, pred):.3f}")
    m3.metric("F1", f"{f1_score(preds.y_true, pred):.3f}")
    m4.metric("Customers flagged", f"{pred.sum():,}")
    cm = confusion_matrix(preds.y_true, pred)
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["Stay", "Churn"], yticklabels=["Stay", "Churn"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    st.pyplot(fig)
    st.info("Lower threshold → catch more churners (higher recall) but contact more customers who "
            "would have stayed (lower precision). Pick the trade-off based on the cost of a retention offer "
            "vs. the value of a lost customer.")

# ---------------- TAB 3: PREDICT ----------------
with tab3:
    st.subheader("Score a single customer")
    c1, c2, c3 = st.columns(3)
    with c1:
        tenure = st.slider("Tenure (months)", 0, 72, 6)
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        internet_s = st.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
        monthly = st.number_input("Monthly charges ($)", 18.0, 130.0, 85.0)
    with c2:
        payment = st.selectbox("Payment method", ["Electronic check", "Mailed check",
                                                  "Bank transfer (automatic)", "Credit card (automatic)"])
        paperless = st.selectbox("Paperless billing", ["Yes", "No"])
        senior = st.selectbox("Senior citizen", [0, 1])
        gender = st.selectbox("Gender", ["Female", "Male"])
    with c3:
        partner = st.selectbox("Partner", ["No", "Yes"])
        dependents = st.selectbox("Dependents", ["No", "Yes"])
        phone = st.selectbox("Phone service", ["Yes", "No"])
        sec = st.selectbox("Online security", ["No", "Yes"])
        backup = st.selectbox("Online backup", ["No", "Yes"])
        tech = st.selectbox("Tech support", ["No", "Yes"])
        tv = st.selectbox("Streaming TV", ["No", "Yes"])
        movies = st.selectbox("Streaming movies", ["No", "Yes"])
        devprot = st.selectbox("Device protection", ["No", "Yes"])
        lines = st.selectbox("Multiple lines", ["No", "Yes"])

    nis = "No internet service"
    row = pd.DataFrame([dict(
        gender=gender, SeniorCitizen=senior, Partner=partner, Dependents=dependents, tenure=tenure,
        PhoneService=phone, InternetService=internet_s, Contract=contract, PaperlessBilling=paperless,
        PaymentMethod=payment, MonthlyCharges=monthly, TotalCharges=monthly * tenure,
        OnlineSecurity=nis if internet_s == "No" else sec, OnlineBackup=nis if internet_s == "No" else backup,
        TechSupport=nis if internet_s == "No" else tech, StreamingTV=nis if internet_s == "No" else tv,
        StreamingMovies=nis if internet_s == "No" else movies,
        DeviceProtection=nis if internet_s == "No" else devprot,
        MultipleLines="No phone service" if phone == "No" else lines)])
    prob = model.predict_proba(engineer(row)[FEATURES])[0, 1]

    st.metric("Churn probability", f"{prob:.1%}")
    st.progress(float(prob))
    if prob >= 0.6:
        st.error("🔴 High risk — consider a retention offer (e.g. discount for a 1-year contract).")
    elif prob >= 0.35:
        st.warning("🟠 Medium risk — monitor and consider proactive outreach.")
    else:
        st.success("🟢 Low risk.")

# ---------------- TAB 4: AT-RISK ----------------
with tab4:
    st.subheader("Highest-risk customers in the test set")
    n = st.slider("Show top N", 10, 200, 25, 5)
    cols = ["customerID", "churn_probability", "Contract", "tenure", "MonthlyCharges",
            "InternetService", "PaymentMethod", "Churn"]
    top = scored.sort_values("churn_probability", ascending=False).head(n)[cols].rename(columns={"Churn": "actually_churned"})
    st.dataframe(top.style.format({"churn_probability": "{:.1%}", "MonthlyCharges": "${:.2f}"}),
                 width="stretch", hide_index=True)
    st.download_button("⬇️ Download as CSV", top.to_csv(index=False), "at_risk_customers.csv", "text/csv")
    st.caption(f"Monthly revenue at risk in this list: **${top.MonthlyCharges.sum():,.0f}**")
