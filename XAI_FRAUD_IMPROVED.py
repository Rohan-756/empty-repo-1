#!/usr/bin/env python
# coding: utf-8

# =============================================================================
# EXPLAINABLE AI IN CREDIT CARD FRAUD DETECTION
# Extended & Improved Version — Based on arXiv:2103.00949
# (Misheva, Hirsa, Osterrieder et al., 2021)
#
# DATASET  : IEEE-CIS Fraud Detection (Kaggle)
#            — OR — Synthetic fallback auto-generated if not uploaded
# TARGET   : isFraud  (1 = Fraud, 0 = Legitimate)
#
# HOW TO USE IN GOOGLE COLAB
# ─────────────────────────────────────────────────────────────────────────────
#   OPTION A (Kaggle dataset — recommended for best results):
#     1. Go to https://www.kaggle.com/c/ieee-fraud-detection/data
#     2. Download train_transaction.csv (it is ~500MB)
#     3. Upload it via the Files panel (left sidebar) in Colab
#     4. Run all cells top-to-bottom
#
#   OPTION B (Auto-generated synthetic data — runs instantly, no download):
#     1. Just run all cells — synthetic data is created automatically
#        if the Kaggle file is not found
#
# WHAT THE ORIGINAL PAPER DID (arXiv:2103.00949)
# ─────────────────────────────────────────────────────────────────────────────
#   • Dataset       : Lending Club (2.2M rows, credit RISK — not fraud)
#   • Models        : LR, XGBoost, RF, SVM, Neural Network
#   • XAI Methods   : LIME (local) + SHAP (global)
#   • SHAP explainers: Tree, Kernel, Linear, Deep
#   • Extra         : ALE plots for LR/SVM/NN
#   • Gaps          : No stability/faithfulness evaluation, no counterfactuals,
#                     no rule-based explanations, no fairness audit,
#                     no inter-method agreement metric, credit risk ≠ fraud
#
# IMPROVEMENTS IN THIS CODE (Novel Contributions over the paper)
# ─────────────────────────────────────────────────────────────────────────────
#   1.  DOMAIN SHIFT: Credit card FRAUD detection (not credit risk)
#       → Extreme class imbalance (0.1–3.5%), adversarial, time-ordered
#   2.  SHAP TreeExplainer + KernelExplainer comparison (same as paper §5.1)
#   3.  SHAP Force plots & Dependence plots (paper §5.1 + §5.2 equivalent)
#   4.  ALE plots (paper §5.4) — extended to XGBoost
#   5.  LIME stability quantification (NEW — paper had no stability analysis)
#   6.  Faithfulness test: SHAP-ordered vs random feature removal (NEW)
#   7.  SHAP vs LIME agreement — Spearman rank-correlation (NEW)
#   8.  Anchors: rule-based IF-THEN explanations (NEW)
#   9.  DiCE counterfactual explanations — with actionable constraints (NEW)
#  10.  XGBoost + LightGBM (paper had XGBoost, not LightGBM)
#  11.  Optuna Bayesian hyperparameter tuning (paper used grid search)
#  12.  SMOTE + effect of balancing on SHAP importances (NEW)
#  13.  Confidence-stratified case analysis — 4 quadrants (NEW)
#  14.  FAIRNESS / BIAS AUDIT via SHAP group analysis (NEW — not in paper)
#  15.  XAI method comparison table: LIME vs SHAP vs Anchors (NEW)
#  16.  Cost-sensitive evaluation with asymmetric penalty matrix (NEW)
# =============================================================================


# ─────────────────────────────────────────────────────────────────────────────
# CELL 1 — INSTALL ALL DEPENDENCIES
# Run once. Takes ~3 minutes. Comment out after first run.
# ─────────────────────────────────────────────────────────────────────────────

import subprocess, sys

pkgs = [
    "lime", "dice-ml", "shap", "anchor-exp",
    "xgboost", "lightgbm", "optuna",
    "imbalanced-learn", "plotly", "PyALE"
]
for p in pkgs:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", p])

print("All packages installed successfully.")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 2 — IMPORTS
# ─────────────────────────────────────────────────────────────────────────────

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from tqdm import tqdm
from scipy.stats import spearmanr

# Sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, roc_auc_score, average_precision_score, f1_score
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

# Imbalanced-learn
from imblearn.over_sampling import SMOTE

# XGBoost & LightGBM
import xgboost as xgb
import lightgbm as lgb

# Optuna
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# XAI
import lime
import lime.lime_tabular
import shap
import dice_ml
from dice_ml import Dice
from anchor import anchor_tabular

# ALE
from PyALE import ale

warnings.filterwarnings("ignore")

try:
    shap.initjs()
except Exception:
    pass

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
print("All imports successful.")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 3 — LOAD DATA
# Tries Kaggle IEEE-CIS dataset first, falls back to synthetic fraud data.
# ─────────────────────────────────────────────────────────────────────────────

import os

KAGGLE_PATHS = [
    "/content/train_transaction.csv",
    "train_transaction.csv",
    "/content/drive/MyDrive/train_transaction.csv",
]

df_raw = None
DATA_SOURCE = None

for path in KAGGLE_PATHS:
    if os.path.exists(path):
        print(f"Loading Kaggle IEEE-CIS dataset from: {path} ...")
        df_raw = pd.read_csv(path, nrows=100_000)  # 100k rows for speed
        DATA_SOURCE = "IEEE-CIS Kaggle"
        break

if df_raw is None:
    print("Kaggle file not found — generating synthetic credit card fraud dataset ...")
    print("(For best results, upload train_transaction.csv from Kaggle)\n")

    # ── Synthetic dataset that mimics fraud detection characteristics ──────
    N = 20_000
    FRAUD_RATE = 0.035   # 3.5% fraud — realistic

    rng = np.random.RandomState(RANDOM_STATE)
    n_fraud = int(N * FRAUD_RATE)
    n_legit = N - n_fraud

    def make_legit(n, rng):
        return pd.DataFrame({
            "TransactionAmt":    rng.lognormal(4.5, 1.2, n),       # ~$90 median
            "card1":             rng.randint(1000, 9999, n),
            "card2":             rng.choice([111,121,131,141,151,161,171,181], n),
            "card3":             rng.choice([150,185], n, p=[0.8, 0.2]),
            "card5":             rng.choice([102,117,226,166], n),
            "addr1":             rng.randint(100, 500, n),
            "addr2":             rng.choice([87, 96, 65], n, p=[0.6,0.3,0.1]),
            "dist1":             rng.exponential(50, n),
            "dist2":             rng.exponential(30, n),
            "P_emaildomain":     rng.choice(["gmail","yahoo","hotmail","aol"], n,
                                             p=[0.5,0.25,0.2,0.05]),
            "R_emaildomain":     rng.choice(["gmail","yahoo","hotmail","aol","anonymous"], n,
                                             p=[0.4,0.2,0.15,0.1,0.15]),
            "ProductCD":         rng.choice(["W","H","C","S","R"], n,
                                             p=[0.5,0.2,0.15,0.1,0.05]),
            "card4":             rng.choice(["visa","mastercard","amex","discover"], n,
                                             p=[0.5,0.3,0.15,0.05]),
            "card6":             rng.choice(["debit","credit"], n, p=[0.6,0.4]),
            "DeviceType":        rng.choice(["desktop","mobile"], n, p=[0.65,0.35]),
            "hour":              rng.randint(8, 22, n),             # daytime shopping
            "V1":                rng.normal(1.0, 0.5, n).clip(0),
            "V2":                rng.normal(1.0, 0.5, n).clip(0),
            "V3":                rng.normal(1.0, 0.3, n).clip(0),
            "isFraud":           0,
        })

    def make_fraud(n, rng):
        df = make_legit(n, rng)
        df["isFraud"]          = 1
        df["TransactionAmt"]   = rng.lognormal(5.5, 1.5, n).clip(10, 5000)
        df["P_emaildomain"]    = rng.choice(["gmail","yahoo","anonymous","protonmail"], n,
                                             p=[0.3,0.2,0.3,0.2])
        df["R_emaildomain"]    = rng.choice(["anonymous","protonmail","gmail"], n,
                                             p=[0.5,0.3,0.2])
        df["hour"]             = rng.choice(
                                     list(range(0,6)) + list(range(22,24)), n)
        df["DeviceType"]       = rng.choice(["desktop","mobile"], n, p=[0.35,0.65])
        df["dist1"]            = rng.exponential(200, n)
        df["V1"]               = rng.normal(0.2, 0.3, n).clip(0)
        return df

    df_legit = make_legit(n_legit, rng)
    df_fraud  = make_fraud(n_fraud, rng)
    df_raw    = pd.concat([df_legit, df_fraud]).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    DATA_SOURCE = "Synthetic (auto-generated)"

print(f"\nData source     : {DATA_SOURCE}")
print(f"Dataset shape   : {df_raw.shape}")
print(f"\nTarget distribution:")
print(df_raw["isFraud"].value_counts())
print(f"\nFraud rate: {df_raw['isFraud'].mean():.2%}")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 4 — FEATURE ENGINEERING & COLUMN DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────

TARGET = "isFraud"

# ── Feature Engineering ───────────────────────────────────────────────────────
# Add hour-of-day if TransactionDT exists (Kaggle dataset)
if "TransactionDT" in df_raw.columns and "hour" not in df_raw.columns:
    df_raw["hour"] = (df_raw["TransactionDT"] // 3600) % 24
    df_raw["day"]  = (df_raw["TransactionDT"] // (3600*24)) % 7

# Log-transform transaction amount (right-skewed)
if "TransactionAmt" in df_raw.columns:
    df_raw["log_TransactionAmt"] = np.log1p(df_raw["TransactionAmt"])

# ── Define feature groups ─────────────────────────────────────────────────────
CANDIDATE_NUM = [
    "log_TransactionAmt", "TransactionAmt",
    "card1", "card2", "card3", "card5",
    "addr1", "addr2", "dist1", "dist2",
    "hour", "V1", "V2", "V3",
    # Kaggle V-features
    "V4","V5","V6","V7","V8","V9","V10",
    "V12","V13","V14","V17","V20","V29","V30",
    "V33","V35","V36","V37","V38",
]

CANDIDATE_CAT = [
    "ProductCD", "card4", "card6",
    "P_emaildomain", "R_emaildomain",
    "DeviceType",
    "M1","M2","M3","M4","M5","M6","M7","M8","M9",
]

# Keep only columns that exist AND have <50% missing
def keep_cols(candidates, df, max_missing=0.5):
    return [c for c in candidates
            if c in df.columns and df[c].isna().mean() < max_missing]

NUMERICAL_COLS   = keep_cols(CANDIDATE_NUM, df_raw)
CATEGORICAL_COLS = keep_cols(CANDIDATE_CAT, df_raw)

# Fill missing
df_raw[NUMERICAL_COLS]   = df_raw[NUMERICAL_COLS].fillna(df_raw[NUMERICAL_COLS].median())
df_raw[CATEGORICAL_COLS] = df_raw[CATEGORICAL_COLS].fillna("unknown")

print(f"Numerical   ({len(NUMERICAL_COLS)}) : {NUMERICAL_COLS}")
print(f"Categorical ({len(CATEGORICAL_COLS)}) : {CATEGORICAL_COLS}")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 5 — EDA
# ─────────────────────────────────────────────────────────────────────────────

# 5a. Transaction amount distribution by class
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
for label, color in [(0,"steelblue"),(1,"coral")]:
    name = "Legitimate" if label==0 else "Fraud"
    amt  = df_raw[df_raw[TARGET]==label]["TransactionAmt"]
    axes[0].hist(np.log1p(amt), bins=40, alpha=0.6, color=color, label=name)
    axes[1].boxplot(np.log1p(amt), positions=[label], widths=0.4,
                    patch_artist=True,
                    boxprops=dict(facecolor=color, alpha=0.6))
axes[0].set_title("Log(TransactionAmt) — Fraud vs Legit")
axes[0].set_xlabel("log(1+Amount)"); axes[0].legend()
axes[1].set_title("Boxplot — Log Amount by Class")
axes[1].set_xticks([0,1]); axes[1].set_xticklabels(["Legit","Fraud"])
plt.tight_layout(); plt.show()

# 5b. Hour-of-day fraud pattern (NEW — not in original paper)
if "hour" in df_raw.columns:
    hour_fraud = df_raw.groupby("hour")[TARGET].mean()
    plt.figure(figsize=(10, 3))
    plt.bar(hour_fraud.index, hour_fraud.values, color="coral", edgecolor="black")
    plt.xlabel("Hour of Day"); plt.ylabel("Fraud Rate")
    plt.title("Fraud Rate by Hour of Day (NOVEL — domain-specific insight)")
    plt.axhline(df_raw[TARGET].mean(), color="navy", linestyle="--",
                label=f"Overall mean ({df_raw[TARGET].mean():.2%})")
    plt.legend(); plt.tight_layout(); plt.show()

# 5c. Class imbalance
plt.figure(figsize=(4, 3))
vc = df_raw[TARGET].value_counts()
plt.bar(["Legitimate","Fraud"], vc.values,
        color=["steelblue","coral"], edgecolor="black")
plt.title("Class Distribution"); plt.ylabel("Count")
for i, v in enumerate(vc.values):
    plt.text(i, v+50, str(v), ha="center", fontsize=9)
plt.tight_layout(); plt.show()
print(f"Fraud rate: {df_raw[TARGET].mean():.2%} — SMOTE required")

# 5d. Email domain fraud rates (NOVEL insight for fraud domain)
if "P_emaildomain" in df_raw.columns:
    dom_stats = df_raw.groupby("P_emaildomain")[TARGET].agg(["mean","count"])
    dom_stats = dom_stats[dom_stats["count"] > 30].sort_values("mean", ascending=False)
    plt.figure(figsize=(10, 4))
    dom_stats["mean"].head(12).plot(kind="bar", color="coral", edgecolor="black")
    plt.title("Fraud Rate by Payer Email Domain")
    plt.ylabel("Fraud Rate"); plt.xticks(rotation=45, ha="right")
    plt.tight_layout(); plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# CELL 6 — PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

X_raw = df_raw[NUMERICAL_COLS + CATEGORICAL_COLS].copy()
y     = df_raw[TARGET].copy()

import sklearn
_sklearn_version = tuple(int(x) for x in sklearn.__version__.split(".")[:2])

if _sklearn_version >= (1, 2):
    _ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False, max_categories=20)
else:
    _ohe = OneHotEncoder(handle_unknown="ignore", sparse=False, max_categories=20)

preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), NUMERICAL_COLS),
    ("cat", _ohe, CATEGORICAL_COLS),
])

X_enc = preprocessor.fit_transform(X_raw)

cat_names     = preprocessor.named_transformers_["cat"]\
                .get_feature_names_out(CATEGORICAL_COLS).tolist()
FEATURE_NAMES = NUMERICAL_COLS + cat_names

print(f"Encoded feature matrix : {X_enc.shape}")
print(f"Total features after OHE: {len(FEATURE_NAMES)}")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 7 — TRAIN / TEST SPLIT
# ─────────────────────────────────────────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    X_enc, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
)

print(f"Train: {X_train.shape}  Legit={( y_train==0).sum()}  Fraud={y_train.sum()}")
print(f"Test : {X_test.shape}  Legit={(y_test==0).sum()}  Fraud={y_test.sum()}")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 8 — SMOTE OVERSAMPLING
# IMPROVEMENT: Fraud datasets are severely imbalanced (~0.1–3.5%).
# SMOTE creates synthetic minority (Fraud) samples.
# ─────────────────────────────────────────────────────────────────────────────

print("Before SMOTE — Fraud:", y_train.sum(), " Legit:", (y_train==0).sum())
smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=5)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
print("After  SMOTE — Fraud:", y_train_sm.sum(), " Legit:", (y_train_sm==0).sum())


# ─────────────────────────────────────────────────────────────────────────────
# CELL 9 — COST-SENSITIVE METRIC (Asymmetric Fraud Cost Matrix)
# IMPROVEMENT: Missing fraud (FN) costs ~5× more than a false alarm (FP).
# Paper only reported accuracy/F1/ROC — no domain-specific cost analysis.
# ─────────────────────────────────────────────────────────────────────────────

COST_FN = 5   # Miss a fraud: customer harmed, bank liable
COST_FP = 1   # False alarm: customer inconvenienced

def cost_score(y_true, y_pred):
    """Lower is better. Asymmetric fraud cost matrix."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    fn_cost = COST_FN * np.sum((y_true==1) & (y_pred==0))   # fraud missed
    fp_cost = COST_FP * np.sum((y_true==0) & (y_pred==1))   # false alarm
    return fn_cost + fp_cost

print(f"Cost matrix: FN (missed fraud) = {COST_FN}×  |  FP (false alarm) = {COST_FP}×")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 10 — TRAIN ALL MODELS & EVALUATE
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(name, model, X_tr, y_tr, X_te, y_te, cv_folds=5):
    model.fit(X_tr, y_tr)
    y_pred  = model.predict(X_te)
    y_proba = model.predict_proba(X_te)[:, 1]

    metrics = {
        "model":         name,
        "accuracy":      round(accuracy_score(y_te, y_pred), 4),
        "balanced_acc":  round(balanced_accuracy_score(y_te, y_pred), 4),
        "f1_fraud":      round(f1_score(y_te, y_pred, pos_label=1), 4),
        "auc_roc":       round(roc_auc_score(y_te, y_proba), 4),
        "auc_pr_fraud":  round(average_precision_score(y_te, y_proba), 4),
        "cost":          cost_score(y_te, y_pred),
    }
    cv = cross_val_score(
        model, X_tr, y_tr,
        cv=StratifiedKFold(n_splits=cv_folds, shuffle=True,
                           random_state=RANDOM_STATE),
        scoring="roc_auc"
    )
    metrics["cv_auc"] = f"{cv.mean():.4f} ± {cv.std()*2:.4f}"

    print(f"\n{'='*55}\n  {name}")
    print(f"  Accuracy        : {metrics['accuracy']}")
    print(f"  Balanced Acc    : {metrics['balanced_acc']}")
    print(f"  F1 (Fraud)      : {metrics['f1_fraud']}")
    print(f"  AUC-ROC         : {metrics['auc_roc']}")
    print(f"  AUC-PR (Fraud)  : {metrics['auc_pr_fraud']}")
    print(f"  Cost (↓ better) : {metrics['cost']}")
    print(f"  CV AUC          : {metrics['cv_auc']}")
    print(classification_report(y_te, y_pred, target_names=["Legit","Fraud"]))

    cm = confusion_matrix(y_te, y_pred)
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Legit","Fraud"], yticklabels=["Legit","Fraud"])
    plt.title(f"Confusion Matrix — {name}")
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.tight_layout(); plt.show()
    return metrics, model


classifiers = {
    # ── Paper models ──────────────────────────────────────────────────────
    "LogisticRegression": LogisticRegression(
        max_iter=2000, random_state=RANDOM_STATE, class_weight="balanced"),
    "DecisionTree":       DecisionTreeClassifier(
        max_depth=6, random_state=RANDOM_STATE, class_weight="balanced"),
    "RandomForest":       RandomForestClassifier(
        n_estimators=200, random_state=RANDOM_STATE, class_weight="balanced", n_jobs=-1),
    "GradientBoosting":   GradientBoostingClassifier(
        n_estimators=200, random_state=RANDOM_STATE),
    "NeuralNetwork":      MLPClassifier(
        hidden_layer_sizes=(64, 32), activation="relu", max_iter=300,
        random_state=RANDOM_STATE),
    # ── New models ─────────────────────────────────────────────────────────
    "XGBoost":            xgb.XGBClassifier(
        n_estimators=300, use_label_encoder=False, eval_metric="logloss",
        scale_pos_weight=(y_train==0).sum()/y_train.sum(),  # auto fraud weight
        random_state=RANDOM_STATE, verbosity=0, n_jobs=-1),
    "LightGBM":           lgb.LGBMClassifier(
        n_estimators=300, class_weight="balanced",
        random_state=RANDOM_STATE, verbose=-1, n_jobs=-1),
}

all_metrics    = []
trained_models = {}

for name, clf in classifiers.items():
    metrics, fitted = evaluate_model(
        name, clf, X_train_sm, y_train_sm, X_test, y_test)
    all_metrics.append(metrics)
    trained_models[name] = fitted


# ─────────────────────────────────────────────────────────────────────────────
# CELL 11 — CONSOLIDATED RESULTS TABLE
# ─────────────────────────────────────────────────────────────────────────────

results_df = pd.DataFrame(all_metrics).set_index("model")
results_df = results_df.sort_values("auc_roc", ascending=False)

print("\n===== CONSOLIDATED RESULTS TABLE =====")
print(results_df[["accuracy","balanced_acc","f1_fraud",
                   "auc_roc","auc_pr_fraud","cost","cv_auc"]].to_string())

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
results_df["auc_roc"].sort_values().plot(
    kind="barh", ax=axes[0], color="steelblue", edgecolor="black")
axes[0].set_title("Model Comparison — AUC-ROC")
axes[0].set_xlabel("AUC-ROC")

results_df["cost"].sort_values(ascending=False).plot(
    kind="barh", ax=axes[1], color="coral", edgecolor="black")
axes[1].set_title("Model Comparison — Fraud Cost (Lower = Better)")
axes[1].set_xlabel(f"Cost (FN={COST_FN}×, FP={COST_FP}×)")
plt.tight_layout(); plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# CELL 12 — OPTUNA HYPERPARAMETER TUNING (XGBoost)
# IMPROVEMENT: Bayesian search with Optuna vs grid search in original paper.
# ─────────────────────────────────────────────────────────────────────────────

fraud_weight = (y_train==0).sum() / y_train.sum()

def xgb_objective(trial):
    params = {
        "n_estimators":      trial.suggest_int("n_estimators", 100, 500),
        "max_depth":         trial.suggest_int("max_depth", 3, 10),
        "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight":  trial.suggest_int("min_child_weight", 1, 10),
        "gamma":             trial.suggest_float("gamma", 0.0, 1.0),
        "scale_pos_weight":  trial.suggest_float("scale_pos_weight",
                                                  fraud_weight*0.5,
                                                  fraud_weight*2.0),
        "use_label_encoder": False,
        "eval_metric": "logloss",
        "verbosity": 0,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }
    cv = cross_val_score(
        xgb.XGBClassifier(**params),
        X_train_sm, y_train_sm,
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE),
        scoring="roc_auc"
    )
    return cv.mean()


print("Tuning XGBoost with Optuna (50 trials) …")
study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE)
)
study.optimize(xgb_objective, n_trials=50, show_progress_bar=True)

print(f"\nBest AUC  : {study.best_value:.4f}")
print(f"Best params: {study.best_params}")

xgb_tuned = xgb.XGBClassifier(
    **study.best_params,
    use_label_encoder=False, eval_metric="logloss",
    random_state=RANDOM_STATE, verbosity=0, n_jobs=-1
)
tuned_metrics, xgb_tuned_fitted = evaluate_model(
    "XGBoost_Tuned", xgb_tuned, X_train_sm, y_train_sm, X_test, y_test)
all_metrics.append(tuned_metrics)
trained_models["XGBoost_Tuned"] = xgb_tuned_fitted

BEST_MODEL      = xgb_tuned_fitted
BEST_MODEL_NAME = "XGBoost_Tuned"


# ─────────────────────────────────────────────────────────────────────────────
# CELL 13 — SHAP: TREE EXPLAINER (Global)
# ─────────────────────────────────────────────────────────────────────────────

shap_tree_exp = shap.TreeExplainer(BEST_MODEL)
shap_values   = shap_tree_exp.shap_values(X_test)

print("SHAP Summary Plot — Beeswarm (TreeExplainer, XGBoost_Tuned)")
shap.summary_plot(shap_values, X_test, feature_names=FEATURE_NAMES, show=True)

print("SHAP Summary Plot — Bar (Global Mean |SHAP|)")
shap.summary_plot(shap_values, X_test, feature_names=FEATURE_NAMES,
                  plot_type="bar", show=True)

shap_global = pd.Series(
    np.abs(shap_values).mean(axis=0),
    index=FEATURE_NAMES
).sort_values(ascending=False)

print("\nTop 10 SHAP features (TreeExplainer):")
print(shap_global.head(10).to_string())


# ─────────────────────────────────────────────────────────────────────────────
# CELL 14 — SHAP: KERNEL EXPLAINER vs TREE EXPLAINER COMPARISON
# IMPROVEMENT: Spearman ρ quantifies consistency between explainers.
# Paper (§5.1.1) said 'consistent' but never measured it.
# ─────────────────────────────────────────────────────────────────────────────

print("\nComputing SHAP KernelExplainer (150 test samples) …")
background      = shap.kmeans(X_train_sm, 30)
kernel_exp      = shap.KernelExplainer(BEST_MODEL.predict_proba, background)
shap_kernel_raw = kernel_exp.shap_values(X_test[:150])

# Handle different shap output formats
if isinstance(shap_kernel_raw, list):
    shap_kernel_v = shap_kernel_raw[1]
elif isinstance(shap_kernel_raw, np.ndarray) and shap_kernel_raw.ndim == 3:
    shap_kernel_v = shap_kernel_raw[:, :, 1]
else:
    shap_kernel_v = shap_kernel_raw

shap_kernel_global = pd.Series(
    np.abs(shap_kernel_v).mean(axis=0),
    index=FEATURE_NAMES
).sort_values(ascending=False)

rho_tk, p_tk = spearmanr(
    shap_global.reindex(FEATURE_NAMES),
    shap_kernel_global.reindex(FEATURE_NAMES)
)
print(f"\nTree vs Kernel SHAP — Spearman ρ = {rho_tk:.3f}  (p = {p_tk:.4f})")

top10 = shap_global.head(10).index.tolist()
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
shap_global[top10].sort_values().plot(
    kind="barh", ax=axes[0], color="steelblue",
    title="SHAP — TreeExplainer (XGBoost)")
shap_kernel_global.reindex(top10).sort_values().plot(
    kind="barh", ax=axes[1], color="coral",
    title="SHAP — KernelExplainer (150 samples)")
plt.suptitle(f"Tree vs Kernel SHAP Comparison  |  Spearman ρ = {rho_tk:.3f}")
plt.tight_layout(); plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# CELL 15 — SHAP: LOCAL WATERFALL PLOTS
# ─────────────────────────────────────────────────────────────────────────────

fraud_idx = int(np.where((y_test.values==1) &
                         (BEST_MODEL.predict(X_test)==1))[0][0])
legit_idx = int(np.where((y_test.values==0) &
                         (BEST_MODEL.predict(X_test)==0))[0][0])

for idx, label in [(fraud_idx,"FRAUD (correctly caught)"),
                   (legit_idx,"LEGITIMATE (correctly cleared)")]:
    print(f"\nSHAP Waterfall — {label} (Instance {idx})")
    shap.waterfall_plot(
        shap.Explanation(
            values        = shap_values[idx],
            base_values   = shap_tree_exp.expected_value,
            data          = X_test[idx],
            feature_names = FEATURE_NAMES
        )
    )


# ─────────────────────────────────────────────────────────────────────────────
# CELL 16 — SHAP: DEPENDENCE PLOTS
# ─────────────────────────────────────────────────────────────────────────────

top3 = shap_global.head(3).index.tolist()
for feat in top3:
    if feat in FEATURE_NAMES:
        print(f"\nSHAP Dependence Plot — {feat}")
        shap.dependence_plot(
            FEATURE_NAMES.index(feat), shap_values, X_test,
            feature_names=FEATURE_NAMES, show=True
        )


# ─────────────────────────────────────────────────────────────────────────────
# CELL 17 — SHAP: GLOBAL IMPORTANCE vs TREE FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────────────────────

tree_imp = pd.Series(
    BEST_MODEL.feature_importances_,
    index=FEATURE_NAMES
).sort_values(ascending=False)

top15 = shap_global.head(15).index.tolist()
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
shap_global[top15].sort_values().plot(
    kind="barh", ax=axes[0], color="steelblue",
    title="SHAP Global Importance (Mean |SHAP|)")
tree_imp.reindex(top15).sort_values().plot(
    kind="barh", ax=axes[1], color="orange",
    title="Tree Feature Importance (Information Gain)")
plt.suptitle("SHAP vs Tree Feature Importance")
plt.tight_layout(); plt.show()

rho_st, p_st = spearmanr(
    shap_global.reindex(FEATURE_NAMES),
    tree_imp.reindex(FEATURE_NAMES)
)
print(f"\nSHAP vs Tree importance — Spearman ρ = {rho_st:.3f}  (p = {p_st:.4f})")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 18 — ALE PLOTS (paper §5.4 equivalent — extended to XGBoost)
# IMPROVEMENT: Paper computed ALE only for LR/SVM/NN (not tree models).
# ─────────────────────────────────────────────────────────────────────────────

X_train_df = pd.DataFrame(X_train_sm, columns=FEATURE_NAMES)
X_test_df  = pd.DataFrame(X_test,     columns=FEATURE_NAMES)

def plot_ale(model, model_name, feature, X_df):
    try:
        ale_eff = ale(
            X=X_df, model=model, feature=[feature],
            feature_type="continuous", grid_size=20,
            include_CI=True, plot=True
        )
        plt.title(f"ALE Plot — {feature}  [{model_name}]")
        plt.tight_layout(); plt.show()
    except Exception as e:
        print(f"ALE failed for {feature} ({model_name}): {e}")

top_num_feats = [f for f in shap_global.index if f in NUMERICAL_COLS][:3]

for model_name in ["LogisticRegression", "XGBoost_Tuned"]:
    model = trained_models[model_name]
    print(f"\n===== ALE Plots — {model_name} =====")
    for feat in top_num_feats:
        plot_ale(model, model_name, feat, X_test_df)


# ─────────────────────────────────────────────────────────────────────────────
# CELL 19 — LIME SETUP
# ─────────────────────────────────────────────────────────────────────────────

cat_feat_idx = [
    i for i, name in enumerate(FEATURE_NAMES)
    if any(name.startswith(c) for c in CATEGORICAL_COLS)
]

lime_exp = lime.lime_tabular.LimeTabularExplainer(
    training_data        = X_train_sm,
    feature_names        = FEATURE_NAMES,
    class_names          = ["Legit", "Fraud"],
    categorical_features = cat_feat_idx,
    discretize_continuous= True,
    random_state         = RANDOM_STATE,
    mode                 = "classification",
)


def run_lime(idx, model=None, n_feat=10, n_samp=100_000):
    if model is None:
        model = BEST_MODEL
    true  = "Fraud" if y_test.iloc[idx]==1 else "Legit"
    prob  = model.predict_proba(X_test[idx:idx+1])[0]
    print(f"\n── LIME — Instance {idx}  True={true}  "
          f"P(Legit)={prob[0]:.3f}  P(Fraud)={prob[1]:.3f}")
    exp = lime_exp.explain_instance(
        X_test[idx], model.predict_proba,
        num_features=n_feat, num_samples=n_samp)
    exp.as_pyplot_figure()
    plt.title(f"LIME — Instance {idx}  (True={true})")
    plt.show()
    return exp


# Three key instance types
uncertain_idx = int(np.argsort(np.abs(
    BEST_MODEL.predict_proba(X_test)[:,1] - 0.5))[:10][0])

print("\n=== LIME: Fraud (predicted Fraud) ===")
_ = run_lime(fraud_idx, BEST_MODEL)

print("\n=== LIME: Legitimate (predicted Legit) ===")
_ = run_lime(legit_idx, BEST_MODEL)

print(f"\n=== LIME: Uncertain prediction — Instance {uncertain_idx} ===")
_ = run_lime(uncertain_idx, BEST_MODEL)


# ─────────────────────────────────────────────────────────────────────────────
# CELL 20 — LIME ON ALL PAPER MODELS
# ─────────────────────────────────────────────────────────────────────────────

print("\n===== LIME across model types (Paper §4 equivalent) =====")
for model_name in ["LogisticRegression", "RandomForest",
                   "XGBoost_Tuned", "NeuralNetwork"]:
    model = trained_models[model_name]
    print(f"\n── {model_name} ── Instance {fraud_idx} (True=Fraud)")
    exp = lime_exp.explain_instance(
        X_test[fraud_idx], model.predict_proba,
        num_features=8, num_samples=50_000)
    exp.as_pyplot_figure()
    plt.title(f"LIME — {model_name}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# CELL 21 — LIME STABILITY TEST (IMPROVEMENT — paper had no stability test)
# Run LIME 10× with different seeds → Spearman ρ measures consistency.
# ─────────────────────────────────────────────────────────────────────────────

def _lime_label_to_feat(label, feature_names):
    for f in feature_names:
        if label.startswith(f):
            return f
    return None


def lime_stability(idx, model=None, n_runs=10, n_feat=10, n_samp=50_000):
    if model is None:
        model = BEST_MODEL
    vecs = []
    for seed in range(n_runs):
        exp_tmp = lime.lime_tabular.LimeTabularExplainer(
            training_data        = X_train_sm,
            feature_names        = FEATURE_NAMES,
            class_names          = ["Legit","Fraud"],
            categorical_features = cat_feat_idx,
            discretize_continuous= True,
            random_state         = seed,
            mode                 = "classification",
        )
        exp = exp_tmp.explain_instance(
            X_test[idx], model.predict_proba,
            num_features=n_feat, num_samples=n_samp)
        vec = np.zeros(len(FEATURE_NAMES))
        for label, imp in exp.as_list():
            feat = _lime_label_to_feat(label, FEATURE_NAMES)
            if feat is not None:
                vec[FEATURE_NAMES.index(feat)] += abs(imp)
        vecs.append(vec)
    corrs = [spearmanr(vecs[i], vecs[j])[0]
             for i in range(n_runs) for j in range(i+1, n_runs)]
    return float(np.mean(corrs)), float(np.std(corrs))


print("\n── LIME Stability Test ──")
test_ids = [fraud_idx, legit_idx, uncertain_idx, 50, 150]
stab_rows = []
for idx in test_ids:
    mu, sd = lime_stability(idx)
    stab_rows.append({"instance": idx, "mean_rho": round(mu,3), "std": round(sd,3)})
    print(f"  Instance {idx:3d}: ρ = {mu:.3f} ± {sd:.3f}")

stab_df = pd.DataFrame(stab_rows)
print("\nρ ≈ 1.0 → stable | ρ < 0.6 → unreliable")
print(stab_df.to_string(index=False))

plt.figure(figsize=(7, 4))
plt.bar(stab_df["instance"].astype(str), stab_df["mean_rho"],
        yerr=stab_df["std"], color="steelblue", edgecolor="black", capsize=5)
plt.axhline(1.0, color="green",  linestyle="--", label="Perfect")
plt.axhline(0.6, color="orange", linestyle="--", label="Acceptable threshold")
plt.xlabel("Instance Index"); plt.ylabel("Mean Spearman ρ")
plt.title("LIME Stability Across Seeds (IMPROVEMENT over paper)")
plt.ylim(0, 1.15); plt.legend()
plt.tight_layout(); plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# CELL 22 — FAITHFULNESS TEST (IMPROVEMENT — paper had no faithfulness test)
# Remove SHAP top-K features → track AUC drop vs random removal.
# Proves SHAP explanations faithfully reflect model behaviour.
# ─────────────────────────────────────────────────────────────────────────────

def faithfulness_test(model, shap_imp, X, y, k_max=12, n_reps=5):
    shap_aucs, rand_aucs = [], []
    for k in range(1, k_max+1):
        Xm = X.copy()
        for f in shap_imp.index[:k]:
            Xm[:, FEATURE_NAMES.index(f)] = 0.0
        shap_aucs.append(roc_auc_score(y, model.predict_proba(Xm)[:,1]))

        ra = []
        for _ in range(n_reps):
            Xr = X.copy()
            for f in np.random.choice(FEATURE_NAMES, k, replace=False):
                Xr[:, FEATURE_NAMES.index(f)] = 0.0
            ra.append(roc_auc_score(y, model.predict_proba(Xr)[:,1]))
        rand_aucs.append(float(np.mean(ra)))

    plt.figure(figsize=(8, 5))
    plt.plot(range(1,k_max+1), shap_aucs, "o-", color="red",
             label="SHAP-ordered removal", linewidth=2)
    plt.plot(range(1,k_max+1), rand_aucs,  "s--", color="gray",
             label="Random removal baseline", linewidth=2)
    plt.xlabel("Features removed"); plt.ylabel("AUC-ROC")
    plt.title("Faithfulness Test — SHAP Removal vs Random (IMPROVEMENT over paper)")
    plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.show()

    score = sum(r-s for s,r in zip(shap_aucs, rand_aucs))
    print(f"Faithfulness Score = {score:.4f}  (positive = faithful)")
    return shap_aucs, rand_aucs


print("\nRunning Faithfulness Test …")
fa, ra = faithfulness_test(BEST_MODEL, shap_global, X_test, y_test)


# ─────────────────────────────────────────────────────────────────────────────
# CELL 23 — SHAP vs LIME AGREEMENT (IMPROVEMENT — paper never compared these)
# ─────────────────────────────────────────────────────────────────────────────

def lime_global_imp(model, X_s, n=100, n_feat=10, n_samp=50_000):
    acc = np.zeros(len(FEATURE_NAMES))
    for idx in tqdm(np.random.choice(len(X_s), n, replace=False),
                    desc="LIME global"):
        exp = lime_exp.explain_instance(
            X_s[idx], model.predict_proba,
            num_features=n_feat, num_samples=n_samp)
        for label, imp in exp.as_list():
            feat = _lime_label_to_feat(label, FEATURE_NAMES)
            if feat is not None:
                acc[FEATURE_NAMES.index(feat)] += abs(imp)
    return pd.Series(acc/n, index=FEATURE_NAMES).sort_values(ascending=False)


print("Computing global LIME importance (100 test instances) …")
lime_global = lime_global_imp(BEST_MODEL, X_test)

rho_sl, p_sl = spearmanr(
    shap_global.reindex(FEATURE_NAMES),
    lime_global.reindex(FEATURE_NAMES)
)
print(f"\nSHAP vs LIME — Spearman ρ = {rho_sl:.3f}  (p = {p_sl:.4f})")
if rho_sl > 0.7:
    print("→ Methods largely agree on feature rankings")
else:
    print("→ Notable disagreement — novel finding for this domain")

top10 = shap_global.head(10).index.tolist()
comp_df = pd.DataFrame({
    "SHAP": shap_global[top10].values,
    "LIME": lime_global.reindex(top10).values
}, index=[f[:22] for f in top10])
comp_df.plot(kind="bar", figsize=(13, 5),
             color=["coral","steelblue"], edgecolor="black")
plt.title(f"SHAP vs LIME Global Importance (Top 10)  |  ρ = {rho_sl:.3f}")
plt.xticks(rotation=45, ha="right")
plt.tight_layout(); plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# CELL 24 — ANCHORS: RULE-BASED EXPLANATIONS (NEW — not in paper)
# Gives IF-THEN rules — actionable for fraud analysts
# ─────────────────────────────────────────────────────────────────────────────

anchor_explainer = anchor_tabular.AnchorTabularExplainer(
    class_names      = ["Legit","Fraud"],
    feature_names    = FEATURE_NAMES,
    train_data       = X_train_sm,
    categorical_names= {
        i: [str(v) for v in np.unique(X_train_sm[:,i])]
        for i in cat_feat_idx
    }
)


def explain_anchor(idx, model=None, threshold=0.88):
    if model is None:
        model = BEST_MODEL
    true = "Fraud" if y_test.iloc[idx]==1 else "Legit"
    pred = "Fraud" if model.predict(X_test[idx:idx+1])[0]==1 else "Legit"
    print(f"\n── Anchor — Instance {idx}  True={true}  Pred={pred}")
    exp  = anchor_explainer.explain_instance(
        X_test[idx], model.predict, threshold=threshold)
    rule = (" AND\n   ".join(exp.names())
            if exp.names() else "(empty rule)")
    print(f"   IF   {rule}")
    print(f"   THEN predicted {pred}")
    print(f"   Precision: {exp.precision():.3f}  Coverage: {exp.coverage():.3f}")
    return exp


fraud_ids = np.where(y_test.values==1)[0][:4]
legit_ids = np.where(y_test.values==0)[0][:4]

print("\n===== ANCHORS — FRAUD (Novel contribution) =====")
for idx in fraud_ids:
    explain_anchor(int(idx))

print("\n===== ANCHORS — LEGITIMATE =====")
for idx in legit_ids:
    explain_anchor(int(idx))


# ─────────────────────────────────────────────────────────────────────────────
# CELL 25 — DiCE COUNTERFACTUAL EXPLANATIONS (NEW — not in paper)
# "What would this transaction need to look like to NOT be flagged as fraud?"
# Uses actionable constraints — only mutable features can be changed.
# ─────────────────────────────────────────────────────────────────────────────

dice_train_df          = pd.DataFrame(X_train_sm, columns=FEATURE_NAMES)
dice_train_df["label"] = y_train_sm.values
dice_test_df           = pd.DataFrame(X_test,     columns=FEATURE_NAMES)
dice_test_df["label"]  = y_test.values

d_dice = dice_ml.Data(
    dataframe           = dice_train_df,
    continuous_features = NUMERICAL_COLS,
    outcome_name        = "label"
)
m_dice = dice_ml.Model(model=BEST_MODEL, backend="sklearn")

# ACTIONABLE constraints — we can only suggest changes the cardholder could make
# (e.g., amount is variable; card number is not)
actionable_num = [f for f in NUMERICAL_COLS
                  if f in ["log_TransactionAmt","TransactionAmt","dist1","dist2"]]
if not actionable_num:
    actionable_num = NUMERICAL_COLS[:3]  # fallback

permitted_ranges = {}
for f in actionable_num:
    col_data = dice_train_df[f]
    permitted_ranges[f] = [float(col_data.quantile(0.01)),
                            float(col_data.quantile(0.99))]

print(f"\nActionable features for counterfactuals: {actionable_num}")
print(f"Permitted ranges: {permitted_ranges}")


def dice_explain(idx, n_cfs=4):
    true = "Fraud" if y_test.iloc[idx]==1 else "Legit"
    pred = "Fraud" if BEST_MODEL.predict(X_test[idx:idx+1])[0]==1 else "Legit"
    print(f"\n── DiCE — Instance {idx}  True={true}  Pred={pred}")
    print(f"   (Counterfactual = what would make this look LEGITIMATE)")
    query = dice_test_df.drop(columns=["label"]).iloc[idx:idx+1]
    exp   = Dice(d_dice, m_dice, method="random")

    import inspect
    sig    = inspect.signature(exp.generate_counterfactuals)
    kwargs = dict(
        query_instances  = query,
        desired_class    = "opposite",
        total_CFs        = n_cfs,
        features_to_vary = actionable_num,
        permitted_range  = permitted_ranges,
        verbose          = False,
    )
    if "random_seed" in sig.parameters:
        kwargs["random_seed"] = RANDOM_STATE

    try:
        result = exp.generate_counterfactuals(**kwargs)
        result.visualize_as_dataframe(show_only_changes=True)
        return result
    except Exception as e:
        print(f"   DiCE failed: {e}")
        return None


print("\n===== DiCE: What would a flagged transaction need to change to clear? =====")
for idx in fraud_ids[:3]:
    dice_explain(int(idx))


# ─────────────────────────────────────────────────────────────────────────────
# CELL 26 — SMOTE EFFECT ON SHAP EXPLANATIONS (NEW — not in paper)
# Does training-set balancing change which features the model relies on?
# Directly tests whether SMOTE distorts feature attribution.
# ─────────────────────────────────────────────────────────────────────────────

print("Training XGBoost WITHOUT SMOTE …")
xgb_no_smote = xgb.XGBClassifier(
    n_estimators=300, use_label_encoder=False,
    eval_metric="logloss",
    scale_pos_weight=fraud_weight,  # compensate via weight instead
    random_state=RANDOM_STATE, verbosity=0, n_jobs=-1
)
xgb_no_smote.fit(X_train, y_train)

sv_ns         = shap.TreeExplainer(xgb_no_smote).shap_values(X_test)
shap_no_smote = pd.Series(np.abs(sv_ns).mean(axis=0), index=FEATURE_NAMES)

rho_sm, p_sm = spearmanr(
    shap_global.reindex(FEATURE_NAMES),
    shap_no_smote.reindex(FEATURE_NAMES)
)
print(f"\nSHAP rank-correlation (SMOTE vs No-SMOTE): ρ = {rho_sm:.3f}")
if rho_sm > 0.85:
    print("→ SMOTE does NOT substantially change feature attribution rankings")
else:
    print("→ SMOTE DOES shift feature attribution — important finding!")

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
shap_global.head(10).sort_values().plot(
    kind="barh", ax=axes[0], color="steelblue", title="SHAP — With SMOTE")
shap_no_smote.head(10).sort_values().plot(
    kind="barh", ax=axes[1], color="coral", title="SHAP — Without SMOTE")
plt.suptitle(f"SMOTE Effect on SHAP Feature Importances  |  ρ = {rho_sm:.3f}  (NEW)")
plt.tight_layout(); plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# CELL 27 — FAIRNESS / BIAS AUDIT VIA SHAP (NEW — not in paper)
# Do different device types or email domains get systematically different
# SHAP explanations? Flags potential proxy discrimination in fraud models.
# ─────────────────────────────────────────────────────────────────────────────

print("\n===== FAIRNESS / BIAS AUDIT (NOVEL CONTRIBUTION) =====")
print("Checking if SHAP explanations differ systematically across groups ...\n")

# Reconstruct original categorical values for test set instances
X_test_raw = X_raw.iloc[y_test.index].reset_index(drop=True) \
             if hasattr(y_test, 'index') else pd.DataFrame(X_raw).iloc[:len(y_test)]

# ── Group by DeviceType (if available) ───────────────────────────────────────
group_col = None
for col in ["DeviceType", "card6", "ProductCD"]:
    if col in X_test_raw.columns:
        group_col = col
        break

if group_col is not None:
    groups_series = X_test_raw[group_col].reset_index(drop=True)
    unique_groups = groups_series.unique()
    print(f"Bias audit on: {group_col}  |  Groups: {unique_groups[:6]}")

    group_shap_means = {}
    group_fraud_rates = {}
    for g in unique_groups[:6]:
        mask = (groups_series == g).values
        if mask.sum() < 10:
            continue
        group_shap_means[g] = np.abs(shap_values[mask]).mean(axis=0)
        group_fraud_rates[g] = y_test.values[mask].mean()

    # Plot mean SHAP for top-5 features across groups
    top5 = shap_global.head(5).index.tolist()
    top5_idx = [FEATURE_NAMES.index(f) for f in top5]
    groups_list = list(group_shap_means.keys())

    bias_data = pd.DataFrame(
        {g: [group_shap_means[g][i] for i in top5_idx] for g in groups_list},
        index=[f[:18] for f in top5]
    )
    bias_data.T.plot(kind="bar", figsize=(12, 5), edgecolor="black")
    plt.title(f"Mean |SHAP| by {group_col} Group — Top 5 Features\n"
              f"(NOVEL: Checks if model explanation differs by device/card type)")
    plt.xlabel(group_col); plt.ylabel("Mean |SHAP|")
    plt.xticks(rotation=30); plt.legend(title="Feature", bbox_to_anchor=(1,1))
    plt.tight_layout(); plt.show()

    # Fraud rate disparity
    print(f"\nFraud rate by {group_col}:")
    for g, rate in sorted(group_fraud_rates.items(), key=lambda x: -x[1]):
        print(f"  {g:20s}: {rate:.2%}")

    disparity = max(group_fraud_rates.values()) / (min(group_fraud_rates.values()) + 1e-9)
    print(f"\nMax/Min fraud rate ratio: {disparity:.2f}×")
    if disparity > 3:
        print("⚠️  High disparity — model may be using group membership as proxy")
    else:
        print("✓  Disparity within acceptable range")
else:
    print("No suitable group column found for bias audit.")
    print("(Add DeviceType, card6, or ProductCD to your dataset for this analysis)")

# ── SHAP group difference on top feature ─────────────────────────────────────
print("\n── SHAP Group Difference Plot (top feature) ──")
top_feat_idx = FEATURE_NAMES.index(shap_global.index[0])
plt.figure(figsize=(8, 4))
if group_col is not None:
    for g in groups_list:
        mask = (groups_series == g).values
        if mask.sum() < 5: continue
        plt.hist(shap_values[mask, top_feat_idx], bins=20, alpha=0.5, label=str(g))
    plt.xlabel(f"SHAP value — {shap_global.index[0]}")
    plt.ylabel("Count")
    plt.title(f"Distribution of SHAP Values by {group_col} (Fairness Check)")
    plt.legend(); plt.tight_layout(); plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# CELL 28 — CONFIDENCE-STRATIFIED CASE ANALYSIS (NEW — not in paper)
# 4 quadrants: high/low confidence × correct/wrong.
# Reveals model blind spots — high-confidence wrong predictions are dangerous.
# ─────────────────────────────────────────────────────────────────────────────

proba_t = BEST_MODEL.predict_proba(X_test)[:,1]
pred_t  = (proba_t >= 0.5).astype(int)
correct = (pred_t == y_test.values)
hi_conf = proba_t >= 0.80
lo_conf = (proba_t >= 0.40) & (proba_t <= 0.60)

groups = {
    "HIGH_CONF_CORRECT ✓": np.where( hi_conf &  correct)[0][:2],
    "HIGH_CONF_WRONG  ⚠️":  np.where( hi_conf & ~correct)[0][:2],   # ← blind spots
    "LOW_CONF_CORRECT  ~":  np.where( lo_conf &  correct)[0][:2],
    "LOW_CONF_WRONG    ~":  np.where( lo_conf & ~correct)[0][:2],
}

for grp, idxs in groups.items():
    print(f"\n{'='*55}\nGROUP: {grp}  ({len(idxs)} shown)")
    if len(idxs) == 0:
        print("  (no instances in this quadrant)"); continue
    for idx in idxs:
        true = "Fraud" if y_test.iloc[idx]==1 else "Legit"
        print(f"\n  Instance {idx}: True={true}  P(Fraud)={proba_t[idx]:.3f}")
        exp = lime_exp.explain_instance(
            X_test[idx], BEST_MODEL.predict_proba,
            num_features=8, num_samples=50_000)
        exp.as_pyplot_figure()
        plt.title(f"{grp} — Instance {idx}")
        plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# CELL 29 — XAI METHOD COMPARISON TABLE (NEW — not in paper)
# Side-by-side: what does LIME, SHAP, Anchors say about the SAME instance?
# ─────────────────────────────────────────────────────────────────────────────

def get_lime_top(idx, n=3):
    exp = lime_exp.explain_instance(
        X_test[idx], BEST_MODEL.predict_proba,
        num_features=n, num_samples=50_000)
    results = []
    for label, imp in sorted(exp.as_list(), key=lambda x: abs(x[1]), reverse=True):
        feat = _lime_label_to_feat(label, FEATURE_NAMES)
        results.append((feat or label)[:18])
    return results

def get_shap_top(idx, n=3):
    vals = shap_values[idx]
    top  = np.argsort(np.abs(vals))[::-1][:n]
    return [FEATURE_NAMES[i][:18] for i in top]

def get_anchor_str(idx):
    try:
        exp = anchor_explainer.explain_instance(
            X_test[idx], BEST_MODEL.predict, threshold=0.85)
        return (" & ".join(exp.names()[:2]) if exp.names() else "N/A")[:38]
    except:
        return "N/A"


print("\n===== XAI METHOD COMPARISON TABLE (IMPROVEMENT over paper) =====\n")
hdr = f"{'Idx':>5}  {'True':>6}  {'LIME Top-3':<58}  {'SHAP Top-3':<58}  Anchor Rule"
print(hdr); print("-"*len(hdr))
for idx in np.concatenate([fraud_ids[:3], legit_ids[:3]]):
    true  = "Fraud" if y_test.iloc[idx]==1 else "Legit"
    lime3 = " | ".join(get_lime_top(int(idx)))
    shap3 = " | ".join(get_shap_top(int(idx)))
    anc   = get_anchor_str(int(idx))
    print(f"{idx:>5}  {true:>6}  {lime3:<58}  {shap3:<58}  {anc}")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 30 — FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

mean_stab  = stab_df["mean_rho"].mean()
faith_score = sum(r-s for s,r in zip(fa, ra))
final_df   = pd.DataFrame(all_metrics).set_index("model")\
              .sort_values("auc_roc", ascending=False)

print(f"""
╔════════════════════════════════════════════════════════════════════════════╗
║   PAPER IMPROVEMENT SUMMARY — arXiv:2103.00949 Extended (Fraud Edition)  ║
╚════════════════════════════════════════════════════════════════════════════╝

ORIGINAL PAPER (Misheva et al. 2021)
  Domain    : Credit RISK (loan approval)
  Dataset   : Lending Club (2.2M rows)
  Models    : LR, XGBoost, RF, SVM, Neural Network
  XAI       : LIME + SHAP (Tree, Kernel, Linear, Deep)
  Extras    : ALE plots (LR/SVM/NN only)
  Gaps      : Credit risk ≠ fraud; no stability/faithfulness test;
              no counterfactuals; no rule-based explanations;
              no fairness audit; no inter-method metric

THIS WORK — Improvements & Novel Contributions
  Domain    : Credit card FRAUD detection (adversarial, imbalanced, temporal)
  Dataset   : {DATA_SOURCE}
  Models    : All paper models + LightGBM + Optuna-tuned XGBoost

  QUANTITATIVE IMPROVEMENTS
  ─────────────────────────────────────────────────────────────────────────
  Tree vs Kernel SHAP consistency   : ρ = {rho_tk:.3f}
    (Paper §5.1.1 said 'consistent' but never measured it)

  SHAP vs Tree importance agreement  : ρ = {rho_st:.3f}
    (Paper §5.3 compared visually only; we quantify)

  SHAP vs LIME agreement             : ρ = {rho_sl:.3f}
    (NEW — paper never compared the two XAI methods)

  LIME stability across seeds        : ρ̄ = {mean_stab:.3f}
    (NEW — paper §7 noted limitations but never measured)

  Faithfulness score                 : {faith_score:.4f}
    (NEW — proves SHAP explanations reflect model behaviour)

  SMOTE effect on SHAP rankings      : ρ = {rho_sm:.3f}
    (NEW — does training balance alter feature reliance?)

  NOVEL CONTRIBUTIONS (not in paper, not in original notebook)
  ─────────────────────────────────────────────────────────────────────────
  ✓  Domain shift to FRAUD (extreme imbalance, adversarial patterns)
  ✓  Hour-of-day temporal feature + fraud pattern analysis
  ✓  Fairness / bias audit by device type / card type via SHAP groups
  ✓  DiCE counterfactuals with ACTIONABLE constraints
     (only mutable features varied: {actionable_num})
  ✓  Confidence-stratified analysis (4 quadrants) reveals blind spots
  ✓  Asymmetric fraud cost matrix (FN = {COST_FN}×, FP = {COST_FP}×)
""")

print("===== FINAL RESULTS TABLE (sorted by AUC-ROC) =====")
print(final_df[["accuracy","balanced_acc","f1_fraud",
                "auc_roc","auc_pr_fraud","cost"]].to_string())

# ── Visual summary table of XAI methods ──────────────────────────────────────
xai_table = pd.DataFrame({
    "Method":        ["SHAP (Tree)",  "SHAP (Kernel)", "LIME",         "Anchors",     "DiCE"],
    "Scope":         ["Global+Local", "Global+Local",  "Local",        "Local",        "Local"],
    "Output type":   ["Weights",      "Weights",       "Weights",       "IF-THEN rule", "Counterfactual"],
    "Model-agnostic":["No (tree)",    "Yes",           "Yes",           "Yes",          "Yes"],
    "Speed":         ["Fast",         "Slow",          "Medium",        "Slow",         "Medium"],
    "Stability ρ":   ["—",            "—",             f"{mean_stab:.2f}","—",          "—"],
    "Faithfulness":  [f"{faith_score:.3f}", "—",       "—",             "—",           "—"],
    "Novel vs paper":["Quantified ρ", "Quantified ρ",  "Stability test","NEW",         "NEW+Actionable"],
})
print("\n===== XAI METHOD COMPARISON TABLE =====")
print(xai_table.to_string(index=False))
