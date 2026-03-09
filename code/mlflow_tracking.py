# =========================================
# Arrhythmia MLflow Tracking (Fast + Pro)
# =========================================

import os
import pandas as pd
import numpy as np
import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")

mlflow.set_experiment("Arrhythmia_Experiment")
import mlflow.sklearn
import mlflow.xgboost

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_curve, auc

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE


# =========================================
# Load Dataset
# =========================================

print("Loading dataset...")

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "arrhythmia.data")

df = pd.read_csv(DATA_PATH, header=None)

df.rename(columns={df.columns[-1]: "class"}, inplace=True)

# Binary classification
df["class"] = df["class"].apply(lambda x: 0 if x == 1 else 1)

# Replace missing values
df.replace("?", np.nan, inplace=True)

df = df.apply(pd.to_numeric)

# Remove columns with too many NaN
df = df.loc[:, df.isnull().mean() < 0.4]

# Fill missing values
df.fillna(df.median(), inplace=True)

print("Dataset shape:", df.shape)

X = df.drop("class", axis=1)
y = df["class"]


# =========================================
# Train Test Split
# =========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)


# =========================================
# Models
# =========================================

models = {

    "RandomForest": RandomForestClassifier(
        n_estimators=50,
        random_state=42,
        n_jobs=-1
    ),

    "SVM": SVC(
        kernel="rbf",
        probability=True
    ),

    "LogisticRegression": LogisticRegression(
        max_iter=500
    ),

    "XGBoost": XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss"
    )
}


# =========================================
# MLflow Experiment
# =========================================

mlflow.set_experiment("Arrhythmia_Experiment")


best_f1 = 0
best_model_name = ""


# =========================================
# Training Loop
# =========================================

for name, model in models.items():

    print("\nTraining:", name)

    with mlflow.start_run(run_name=name):

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("smote", SMOTE(random_state=42)),
            ("classifier", model)
        ])

        # Train model
        pipeline.fit(X_train, y_train)

        # Predictions
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Cross Validation (fast)
        cv_scores = cross_val_score(
            pipeline,
            X,
            y,
            cv=3,
            scoring="f1"
        )

        cv_mean = cv_scores.mean()

        print(f"{name} → Accuracy: {acc:.4f}, F1: {f1:.4f}")

        # =====================================
        # MLflow Logging
        # =====================================

        mlflow.log_param("model", name)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("cv_f1_mean", cv_mean)

        mlflow.log_param("dataset", "UCI Arrhythmia")
        mlflow.log_param("samples", len(df))
        mlflow.log_param("features", X.shape[1])

        # =====================================
        # Confusion Matrix
        # =====================================

        cm = confusion_matrix(y_test, y_pred)

        plt.figure(figsize=(5,4))

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues"
        )

        plt.title(f"{name} Confusion Matrix")

        mlflow.log_figure(
            plt.gcf(),
            "confusion_matrix.png"
        )

        plt.close()

        # =====================================
        # ROC Curve
        # =====================================

        fpr, tpr, _ = roc_curve(y_test, y_prob)

        roc_auc = auc(fpr, tpr)

        plt.figure()

        plt.plot(fpr, tpr)

        plt.title(f"{name} ROC Curve")

        mlflow.log_metric("roc_auc", roc_auc)

        mlflow.log_figure(
            plt.gcf(),
            "roc_curve.png"
        )

        plt.close()

        # =====================================
        # Feature Importance
        # =====================================

        if name in ["RandomForest", "XGBoost"]:

            importances = model.feature_importances_

            plt.figure(figsize=(8,4))

            plt.bar(
                range(len(importances)),
                importances
            )

            plt.title("Feature Importance")

            mlflow.log_figure(
                plt.gcf(),
                "feature_importance.png"
            )

            plt.close()

        # =====================================
        # Log Model
        # =====================================

        if name == "XGBoost":

            mlflow.xgboost.log_model(
                model,
                "model"
            )

        else:

            mlflow.sklearn.log_model(
                pipeline,
                "model"
            )

        # =====================================
        # Best Model Selection
        # =====================================

        if f1 > best_f1:

            best_f1 = f1
            best_model_name = name


print("\nBest Model:", best_model_name)
print("Best F1 Score:", best_f1)