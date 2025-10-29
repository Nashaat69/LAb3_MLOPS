#!/usr/bin/env python3
"""
tune.py — Hyperparameter tuning with MLflow nested runs
"""
import os
import joblib
import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# --- MLflow setup ---
mlflow.set_experiment("Heart Disease Model Tuning")

# Load datasets
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

target_col = train_df.columns[-1]
X_train = train_df.drop(columns=[target_col])
y_train = train_df[target_col]

X_test = test_df.drop(columns=[target_col])
y_test = test_df[target_col]

# Hyperparameter grid
n_estimators_list = [100, 200]
max_depth_list = [5, 10, None]

# Parent MLflow run
with mlflow.start_run(run_name="rf_tuning_parent") as parent_run:
    best_acc = 0.0
    best_params = None
    best_model = None

    for n_est in n_estimators_list:
        for depth in max_depth_list:
            # Start nested run
            with mlflow.start_run(run_name=f"rf_n{n_est}_d{depth}", nested=True):
                model = RandomForestClassifier(
                    n_estimators=n_est,
                    max_depth=depth,
                    random_state=42
                )
                model.fit(X_train, y_train)

                # Evaluate
                y_pred = model.predict(X_test)
                acc = accuracy_score(y_test, y_pred)

                # Log parameters & metrics
                mlflow.log_param("n_estimators", n_est)
                mlflow.log_param("max_depth", depth)
                mlflow.log_metric("accuracy", acc)

                if acc > best_acc:
                    best_acc = acc
                    best_params = (n_est, depth)
                    best_model = model

    # Log best result to parent run
    mlflow.log_metric("best_accuracy", best_acc)
    mlflow.log_params({"best_n_estimators": best_params[0], "best_max_depth": best_params[1]})

# Save best model
os.makedirs("models", exist_ok=True)
joblib.dump(best_model, "models/best_model.pkl")
print(f"Best model saved to models/best_model.pkl — Accuracy: {best_acc:.4f}")
