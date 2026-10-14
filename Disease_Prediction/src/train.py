from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data" / "heart.csv"
MODELS = BASE / "models"
MODELS.mkdir(exist_ok=True)

df = pd.read_csv(DATA)

# Accept either "target" or "num" as the target column.
if "target" in df.columns:
    target_col = "target"
elif "num" in df.columns:
    target_col = "num"
    # UCI heart disease 'num' can be 0-4: convert to binary.
    df[target_col] = (pd.to_numeric(df[target_col], errors="coerce") > 0).astype(int)
else:
    raise ValueError("Dataset must contain a 'target' or 'num' column.")

# Convert '?' to missing values and keep numeric rows.
df = df.replace("?", pd.NA)
for c in df.columns:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna()

X = df.drop(columns=[target_col])
y = df[target_col].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000))
    ]),
    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(probability=True, random_state=42))
    ]),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, random_state=42
    ),
    "XGBoost": XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9,
        eval_metric="logloss", random_state=42
    )
}

results = {}
best_name, best_model, best_f1 = None, None, -1

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]

    scores = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, prob)
    }
    results[name] = scores
    print(name, {k: round(v, 4) for k, v in scores.items()})

    if scores["f1"] > best_f1:
        best_f1 = scores["f1"]
        best_name, best_model = name, model

joblib.dump(best_model, MODELS / "heart_model.pkl")
with open(MODELS / "metadata.json", "w") as f:
    json.dump({
        "best_model": best_name,
        "features": list(X.columns),
        "metrics": results
    }, f, indent=2)

print(f"\nSaved best model: {best_name}")
print("Run the app with: streamlit run app.py")
