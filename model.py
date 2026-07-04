"""
Bank Term Deposit Subscription Prediction
==========================================
Predicts whether a client subscribes to a term deposit (target: y = yes/no)
using the UCI Bank Marketing dataset (bank-full.csv, ';' separated, 45211 rows).

Usage:
    python bank_term_deposit_model.py --data bank-full.csv

Outputs:
    - EDA summary printed to console
    - Two trained models: WITH duration and WITHOUT duration (realistic)
    - Evaluation metrics (ROC-AUC, PR-AUC, classification report, confusion matrix)
    - Feature importance chart
    - Saved models (.pkl) and a feature-importance PNG in the output folder
"""

import argparse
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, classification_report,
    confusion_matrix, RocCurveDisplay, PrecisionRecallDisplay
)

warnings.filterwarnings("ignore")

NUMERIC_COLS = ["age", "balance", "day", "campaign", "pdays", "previous"]
CATEGORICAL_COLS = ["job", "marital", "education", "default", "housing",
                     "loan", "contact", "month", "poutcome"]
DURATION_COL = "duration"
TARGET = "y"


def load_data(path):
    df = pd.read_csv(path, sep=";")
    # bank-full.csv sometimes has quoted strings; normalize
    df.columns = [c.strip().strip('"') for c in df.columns]
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].str.strip().str.strip('"')
    return df


def basic_eda(df):
    print("=" * 60)
    print("BASIC EDA")
    print("=" * 60)
    print(f"Shape: {df.shape}")
    print("\nTarget distribution:")
    print(df[TARGET].value_counts())
    print(df[TARGET].value_counts(normalize=True).round(3))
    print("\nMissing values (as 'unknown' category, not NaN, in this dataset):")
    print(df.isna().sum().sum(), "true NaNs")
    print("\nNumeric summary:")
    print(df[NUMERIC_COLS + [DURATION_COL]].describe().T)
    print("\npdays = -1 (never contacted before) count:",
          (df["pdays"] == -1).sum())


def engineer_features(df):
    df = df.copy()
    df["was_contacted_before"] = (df["pdays"] != -1).astype(int)
    # keep pdays but cap -1 sentinel effect isn't removed; model can learn combo w/ flag
    return df


def build_pipeline(numeric_cols, categorical_cols, model):
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )
    return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])


def evaluate_model(name, pipe, X_test, y_test):
    proba = pipe.predict_proba(X_test)[:, 1]
    preds = pipe.predict(X_test)
    roc_auc = roc_auc_score(y_test, proba)
    pr_auc = average_precision_score(y_test, proba)

    print(f"\n--- {name} ---")
    print(f"ROC-AUC: {roc_auc:.4f}   PR-AUC: {pr_auc:.4f}")
    print(classification_report(y_test, preds, target_names=["no", "yes"]))
    print("Confusion matrix [rows=actual, cols=predicted] (no, yes):")
    print(confusion_matrix(y_test, preds))
    return roc_auc, pr_auc


def run_experiment(df_train, df_test, feature_cols, numeric_cols, categorical_cols, label, out_dir):
    X_train = df_train[feature_cols]
    y_train = (df_train[TARGET] == "yes").astype(int)
    X_test = df_test[feature_cols]
    y_test = (df_test[TARGET] == "yes").astype(int)

    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_depth=None, class_weight="balanced",
            random_state=42, n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingClassifier(random_state=42),
    }

    results = {}
    best_name, best_pipe, best_auc = None, None, -1

    for name, model in models.items():
        pipe = build_pipeline(numeric_cols, categorical_cols, model)
        pipe.fit(X_train, y_train)
        roc_auc, pr_auc = evaluate_model(f"{label} | {name}", pipe, X_test, y_test)
        results[name] = (roc_auc, pr_auc)
        if roc_auc > best_auc:
            best_name, best_pipe, best_auc = name, pipe, roc_auc

    # Save best model
    model_path = f"{out_dir}/best_model_{label.replace(' ', '_')}.pkl"
    joblib.dump(best_pipe, model_path)
    print(f"\nBest model for '{label}': {best_name} (ROC-AUC={best_auc:.4f}) -> saved to {model_path}")

    # Feature importance (tree models only)
    if best_name in ("RandomForest", "GradientBoosting"):
        ohe = best_pipe.named_steps["preprocess"].named_transformers_["cat"]
        cat_feature_names = ohe.get_feature_names_out(categorical_cols)
        all_feature_names = np.concatenate([numeric_cols, cat_feature_names])
        importances = best_pipe.named_steps["model"].feature_importances_
        imp_df = pd.Series(importances, index=all_feature_names).sort_values(ascending=False).head(15)

        plt.figure(figsize=(8, 6))
        imp_df.sort_values().plot(kind="barh")
        plt.title(f"Top 15 Feature Importances - {label} ({best_name})")
        plt.tight_layout()
        fig_path = f"{out_dir}/feature_importance_{label.replace(' ', '_')}.png"
        plt.savefig(fig_path, dpi=150)
        plt.close()
        print(f"Feature importance chart saved to {fig_path}")

    return results, best_pipe


def main(train_path, test_path, out_dir="."):
    df_train_raw = load_data(train_path)
    df_test_raw = load_data(test_path)

    # IMPORTANT: in this dataset, test.csv (bank.csv) is a 10% random sample DRAWN FROM
    # train.csv (bank-full.csv) -- it is not an independent holdout. Every row in test.csv
    # already exists in train.csv, which causes data leakage (perfect/inflated scores) if
    # used naively as train/test. We combine + de-duplicate, then create our own proper
    # stratified split.
    combined = pd.concat([df_train_raw, df_test_raw], ignore_index=True)
    before = len(combined)
    combined = combined.drop_duplicates().reset_index(drop=True)
    after = len(combined)
    print(f"Combined rows: {before}, after de-duplication: {after} "
          f"({before - after} duplicate rows removed)")

    print("\nFULL (de-duplicated) DATASET")
    basic_eda(combined)

    df_train, df_test = train_test_split(
        combined, test_size=0.2, stratify=combined[TARGET], random_state=42
    )
    print(f"\nNew stratified split -> train: {len(df_train)}, test: {len(df_test)}")

    df_train = engineer_features(df_train)
    df_test = engineer_features(df_test)

    numeric_cols = NUMERIC_COLS + ["was_contacted_before"]
    categorical_cols = CATEGORICAL_COLS

    print("\n" + "=" * 60)
    print("EXPERIMENT 1: WITHOUT duration (realistic, pre-call model)")
    print("=" * 60)
    feature_cols_no_dur = numeric_cols + categorical_cols
    run_experiment(df_train, df_test, feature_cols_no_dur, numeric_cols, categorical_cols,
                   "without_duration", out_dir)

    print("\n" + "=" * 60)
    print("EXPERIMENT 2: WITH duration (upper-bound / post-call analysis)")
    print("=" * 60)
    feature_cols_with_dur = numeric_cols + [DURATION_COL] + categorical_cols
    numeric_cols_dur = numeric_cols + [DURATION_COL]
    run_experiment(df_train, df_test, feature_cols_with_dur, numeric_cols_dur, categorical_cols,
                   "with_duration", out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, help="Path to train.csv")
    parser.add_argument("--test", required=True, help="Path to test.csv")
    parser.add_argument("--out", default=".", help="Output directory for models/plots")
    args = parser.parse_args()
    main(args.train, args.test, args.out)
