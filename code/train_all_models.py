"""
Complete ML Pipeline: Train and Register All Models to MLflow
===============================================================

This script trains 4 different ML models on the arrhythmia dataset:
- Random Forest
- Support Vector Machine (SVM)
- Logistic Regression  
- XGBoost

All models are logged to MLflow with their metrics and artifacts.
"""

import warnings
warnings.filterwarnings("ignore")

import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from datetime import datetime
import json

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, 
    recall_score, roc_auc_score, confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE


print("=" * 70)
print("ARRHYTHMIA ML PIPELINE - TRAINING ALL MODELS")
print("=" * 70)

# ===============================
# 1️⃣ MLflow Configuration
# ===============================

MLFLOW_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "Arrhythmia_Advanced_Experiment"

mlflow.set_tracking_uri(MLFLOW_URI)

# Check if experiment exists, if not create it
try:
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        mlflow.create_experiment(EXPERIMENT_NAME)
    mlflow.set_experiment(EXPERIMENT_NAME)
    print(f"✅ Using experiment: {EXPERIMENT_NAME}")
except Exception as e:
    print(f"⚠️ Warning setting up MLflow: {e}")

# ===============================
# 2️⃣ Load & Prepare Dataset
# ===============================

print("\n📊 Loading dataset...")

data_path = os.path.join("data", "arrhythmia.data")

if not os.path.exists(data_path):
    print(f"❌ Dataset not found at {data_path}")
    print("Please ensure the dataset is in the data/ folder")
    exit(1)

df = pd.read_csv(data_path, header=None)
print(f"   Loaded: {df.shape[0]} records, {df.shape[1]} features")

# Rename last column to class
df.rename(columns={df.columns[-1]: "class"}, inplace=True)

# ===============================
# 3️⃣ Data Cleaning
# ===============================

print("\n🧹 Cleaning data...")

# Replace ? with NaN
df.replace("?", np.nan, inplace=True)

# Convert all columns to numeric
df = df.apply(pd.to_numeric, errors="coerce")

# Remove columns with >40% missing values
initial_cols = len(df.columns)
df = df.loc[:, df.isnull().mean() < 0.4]
print(f"   Removed {initial_cols - len(df.columns)} columns with >40% missing")

# Fill remaining missing values with median
df.fillna(df.median(), inplace=True)

# Binary classification (Normal=0 vs Arrhythmia=1)
df["class"] = df["class"].apply(lambda x: 0 if x == 1 else 1)

print(f"   Final shape: {df.shape}")
print(f"   Class distribution:\n{df['class'].value_counts()}")

# ===============================
# 4️⃣ Split Data
# ===============================

print("\n✂️ Splitting dataset...")

X = df.drop("class", axis=1)
y = df["class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print(f"   Train: {X_train.shape[0]} records")
print(f"   Test: {X_test.shape[0]} records")
print(f"   Features per record: {X_train.shape[1]}")

# ===============================
# 5️⃣ Models Configuration
# ===============================

models_config = {
    "RandomForest": {
        "model": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ),
        "params": {
            "n_estimators": 200,
            "max_depth": 10,
            "algorithm": "RandomForest",
            "type": "Ensemble"
        }
    },
    "SVM": {
        "model": SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            probability=True,
            random_state=42
        ),
        "params": {
            "kernel": "rbf",
            "C": 1.0,
            "algorithm": "SVM",
            "type": "Kernel Method"
        }
    },
    "LogisticRegression": {
        "model": LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
            random_state=42
        ),
        "params": {
            "max_iter": 1000,
            "solver": "lbfgs",
            "algorithm": "LogisticRegression",
            "type": "Linear"
        }
    },
    "XGBoost": {
        "model": XGBClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1
        ),
        "params": {
            "n_estimators": 200,
            "learning_rate": 0.1,
            "max_depth": 5,
            "algorithm": "XGBoost",
            "type": "Gradient Boosting"
        }
    }
}

# ===============================
# 6️⃣ Train & Log All Models
# ===============================

print("\n" + "=" * 70)
print("TRAINING MODELS")
print("=" * 70)

results = {}

for model_name, config in models_config.items():
    print(f"\n🚀 Training {model_name}...")
    print("-" * 70)
    
    try:
        with mlflow.start_run(run_name=model_name) as run:
            
            # Create pipeline
            pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("smote", SMOTE(random_state=42)),
                ("classifier", config["model"])
            ])
            
            # Train
            print(f"   Training {model_name} pipeline...")
            pipeline.fit(X_train, y_train)
            
            # Predict
            y_pred = pipeline.predict(X_test)
            y_pred_proba = pipeline.predict_proba(X_test)[:, 1] if hasattr(config["model"], 'probability') or hasattr(config["model"], 'predict_proba') else y_pred
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            
            try:
                roc_auc = roc_auc_score(y_test, y_pred_proba)
            except:
                roc_auc = 0.0
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            
            # Log parameters
            print(f"   Logging parameters...")
            mlflow.log_params(config["params"])
            mlflow.log_param("model", model_name)
            
            # Log metrics
            print(f"   Logging metrics...")
            metrics = {
                "accuracy": accuracy,
                "f1_score": f1,
                "precision": precision,
                "recall": recall,
                "roc_auc": roc_auc,
                "true_negatives": int(cm[0, 0]),
                "false_positives": int(cm[0, 1]),
                "false_negatives": int(cm[1, 0]),
                "true_positives": int(cm[1, 1])
            }
            
            for metric_name, value in metrics.items():
                mlflow.log_metric(metric_name, value)
            
            # Log model
            print(f"   Logging model artifacts...")
            if model_name == "XGBoost":
                mlflow.xgboost.log_model(
                    config["model"],
                    artifact_path="model",
                    registered_model_name=f"Arrhythmia_{model_name}"
                )
            else:
                mlflow.sklearn.log_model(
                    pipeline,
                    artifact_path="model",
                    registered_model_name=f"Arrhythmia_{model_name}"
                )
            
            # Log dataset info
            dataset_info = {
                "total_records": len(df),
                "train_records": len(X_train),
                "test_records": len(X_test),
                "features": X_train.shape[1],
                "class_distribution": {
                    "class_0": int((y == 0).sum()),
                    "class_1": int((y == 1).sum())
                }
            }
            mlflow.log_dict(dataset_info, "dataset_info.json")
            
            # Store results
            results[model_name] = metrics
            
            print(f"\n✅ {model_name} Training Complete!")
            print(f"   Accuracy:  {accuracy:.4f}")
            print(f"   F1 Score:  {f1:.4f}")
            print(f"   Precision: {precision:.4f}")
            print(f"   Recall:    {recall:.4f}")
            print(f"   ROC-AUC:   {roc_auc:.4f}")
            print(f"   Run ID:    {run.info.run_id}")
    
    except Exception as e:
        print(f"❌ Error training {model_name}: {str(e)}")
        import traceback
        traceback.print_exc()

# ===============================
# 7️⃣ Results Summary
# ===============================

print("\n" + "=" * 70)
print("TRAINING COMPLETE - RESULTS SUMMARY")
print("=" * 70)

if results:
    # Create comparison dataframe
    results_df = pd.DataFrame(results).T
    results_df = results_df.round(4)
    
    print("\n" + results_df.to_string())
    
    # Find best model
    best_model = results_df['accuracy'].idxmax()
    best_score = results_df.loc[best_model, 'accuracy']
    
    print(f"\n🏆 Best Model: {best_model} (Accuracy: {best_score:.4f})")
    
    # Save results
    results_file = "training_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n📁 Results saved to: {results_file}")

print(f"\n📊 View results in MLflow: {MLFLOW_URI}")
print(f"🔗 MLflow Experiment: {EXPERIMENT_NAME}")
print("\n✨ Training pipeline complete!")
print("=" * 70)
