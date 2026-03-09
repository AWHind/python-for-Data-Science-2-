import warnings
warnings.filterwarnings("ignore")

import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE


# ===============================
# 1️⃣ MLflow Setup
# ===============================

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Arrhythmia_Experiment")


# ===============================
# 2️⃣ Load Dataset
# ===============================

print("Loading dataset...")

data_path = os.path.join("data", "arrhythmia.data")

df = pd.read_csv(data_path, header=None)


# Rename last column to class
df.rename(columns={df.columns[-1]: "class"}, inplace=True)


# ===============================
# 3️⃣ Data Cleaning
# ===============================

# Replace ? with NaN
df.replace("?", np.nan, inplace=True)

# Convert all columns to numeric
df = df.apply(pd.to_numeric, errors="coerce")

# Remove columns with >40% missing values
df = df.loc[:, df.isnull().mean() < 0.4]

# Fill remaining missing values with median
df.fillna(df.median(), inplace=True)

# Binary classification (Normal vs Others)
df["class"] = df["class"].apply(lambda x: 0 if x == 1 else 1)


# ===============================
# 4️⃣ Split Data
# ===============================

X = df.drop("class", axis=1)
y = df["class"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)


# ===============================
# 5️⃣ Model Pipeline
# ===============================

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("smote", SMOTE(random_state=42)),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42
    ))
])


# ===============================
# 6️⃣ Train + Log to MLflow
# ===============================

with mlflow.start_run():

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    f1 = f1_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)

    print(f"F1 Score: {f1}")
    print(f"Accuracy: {acc}")

    # Log parameters
    mlflow.log_param("model", "RandomForest")
    mlflow.log_param("n_estimators", 200)
    mlflow.log_param("max_depth", 10)

    # Log metrics
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("accuracy", acc)

    # Log model + register
    mlflow.sklearn.log_model(
        sk_model=pipeline,
        artifact_path="model",
        registered_model_name="Arrhythmia_Model"
    )

print("✅ Model trained and logged successfully!")
