# =========================================
# Week 3 - Advanced MLflow Tracking
# =========================================

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.xgboost

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE


# ===============================
# Load Dataset
# ===============================

df = pd.read_csv("../data/arrhythmia.data", header=None)
df.rename(columns={df.columns[-1]: "class"}, inplace=True)

df["class"] = df["class"].apply(lambda x: 0 if x == 1 else 1)

df.replace("?", np.nan, inplace=True)
df = df.apply(pd.to_numeric)
df = df.loc[:, df.isnull().mean() < 0.4]
df.fillna(df.median(), inplace=True)

X = df.drop("class", axis=1)
y = df["class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# ===============================
# Models
# ===============================

models = {
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
    "SVM": SVC(kernel="rbf", probability=True),
    "LogisticRegression": LogisticRegression(max_iter=500),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss")
}

mlflow.set_experiment("Arrhythmia_Advanced_Experiment")

for name, model in models.items():

    with mlflow.start_run(run_name=name):

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("smote", SMOTE(random_state=42)),
            ("classifier", model)
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Log parameters
        mlflow.log_param("model", name)

        # Log metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)

        # Log model
        if name == "XGBoost":
            mlflow.xgboost.log_model(model, "model")
        else:
            mlflow.sklearn.log_model(pipeline, "model")

        print(f"{name} → Accuracy: {acc:.4f}, F1: {f1:.4f}")
