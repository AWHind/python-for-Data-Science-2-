# =====================================================
# Week 2 - Machine Learning Pipeline
# Project: Arrhythmia Detection
# =====================================================

import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE


# =====================================================
# 0️⃣ Paths Configuration (Robust Handling)
# =====================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "arrhythmia.data")
MODEL_PATH = os.path.join(BASE_DIR, "data", "best_model.pkl")

os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)


# =====================================================
# 1️⃣ Load Dataset
# =====================================================

print("Loading dataset...")

df = pd.read_csv(DATA_PATH, header=None)
df.rename(columns={df.columns[-1]: "class"}, inplace=True)

print("Initial shape:", df.shape)


# =====================================================
# 2️⃣ Target Transformation (Binary Classification)
# =====================================================

# 1 -> 0 (Normal)
# 2-16 -> 1 (Arrhythmia)
df["class"] = df["class"].apply(lambda x: 0 if x == 1 else 1)

print("\nClass distribution:")
print(df["class"].value_counts())


# =====================================================
# 3️⃣ Data Cleaning
# =====================================================

# Replace '?' with NaN
df.replace("?", np.nan, inplace=True)

# Convert to numeric
df = df.apply(pd.to_numeric)

# Remove columns with >40% missing values
df = df.loc[:, df.isnull().mean() < 0.4]

# Fill remaining missing values with median
df.fillna(df.median(), inplace=True)

print("\nShape after cleaning:", df.shape)


# =====================================================
# 4️⃣ Train / Test Split
# =====================================================

X = df.drop("class", axis=1)
y = df["class"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("\nTrain shape:", X_train.shape)
print("Test shape:", X_test.shape)


# =====================================================
# 5️⃣ Build ML Pipeline (Scaler + SMOTE + RF)
# =====================================================

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("smote", SMOTE(random_state=42)),
    ("classifier", RandomForestClassifier(random_state=42))
])


# =====================================================
# 6️⃣ Hyperparameter Tuning (GridSearchCV)
# =====================================================

param_grid = {
    "classifier__n_estimators": [100, 200],
    "classifier__max_depth": [None, 10, 20],
    "classifier__min_samples_split": [2, 5]
}

print("\nTraining model with GridSearch...")

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print("\nBest Parameters Found:")
print(grid_search.best_params_)


# =====================================================
# 7️⃣ Evaluation
# =====================================================

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# =====================================================
# 8️⃣ Save Model
# =====================================================

joblib.dump(best_model, MODEL_PATH)

print(f"\nModel saved successfully in: {MODEL_PATH}")
