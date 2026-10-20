"""Clean -> engineer -> compare models -> save best model + metrics."""
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from features import CATEGORICAL, FEATURES, NUMERIC, clean, engineer

SEED = 42
df = engineer(clean(pd.read_csv("data/churn.csv")))
X, y = df[FEATURES], df["Churn"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

prep = ColumnTransformer([
    ("num", StandardScaler(), NUMERIC),
    ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
])
models = {
    "Baseline (majority)": DummyClassifier(strategy="most_frequent"),
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                            class_weight="balanced", random_state=SEED, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(random_state=SEED),
}

cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
results, fitted, probas = {}, {}, {}
for name, clf in models.items():
    pipe = Pipeline([("prep", prep), ("clf", clf)])
    cv_auc = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="roc_auc").mean()
    pipe.fit(X_tr, y_tr)
    p = pipe.predict_proba(X_te)[:, 1]
    pred = (p >= 0.5).astype(int)
    results[name] = dict(
        accuracy=accuracy_score(y_te, pred),
        precision=precision_score(y_te, pred, zero_division=0),
        recall=recall_score(y_te, pred),
        f1=f1_score(y_te, pred),
        roc_auc=roc_auc_score(y_te, p),
        cv_roc_auc=cv_auc,
    )
    fitted[name], probas[name] = pipe, p
    print(f"{name:22s}", {k: round(v, 3) for k, v in results[name].items()})

best = max((n for n in results if not n.startswith("Baseline")), key=lambda n: results[n]["roc_auc"])
print("\nBest model:", best)

imp = permutation_importance(fitted[best], X_te, y_te, scoring="roc_auc",
                             n_repeats=5, random_state=SEED, n_jobs=-1)
importance = (pd.DataFrame({"feature": FEATURES, "importance": imp.importances_mean})
              .sort_values("importance", ascending=False))

joblib.dump(fitted[best], "models/best_model.joblib")
importance.to_csv("models/feature_importance.csv", index=False)
pd.DataFrame({name: p for name, p in probas.items()}).assign(y_true=y_te.values) \
    .to_csv("models/test_predictions.csv", index=False)
test_df = df.loc[X_te.index].assign(churn_probability=probas[best])
test_df.to_csv("models/test_scored.csv", index=False)
json.dump({"best_model": best, "results": results}, open("models/metrics.json", "w"), indent=2)
