"""
Shared harness for feature-engineering / model experiments on the EV-purchase dataset.

Design goals:
- Every experiment is compared against the SAME fixed holdout split used in the
  notebook's tuned-model cell (test_size=0.2, random_state=42, stratify=y), so
  numbers here are directly comparable to the 0.9410766 holdout score / 0.94083
  Kaggle submission already logged in the repo.
- A 5-fold CV harness is also provided for confirming any promising result before
  it gets treated as "real" -- a single holdout can move +/-0.001 just from noise.
- Baseline model params match the tuned CatBoost config already in eda.ipynb / the
  submission that scored 0.94083 on the public leaderboard, so deltas are attributable
  to the feature/model change being tested, not to a params change.
"""
import time
import json
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier

DATA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.jsonl")

BASE_CATBOOST_PARAMS = dict(
    iterations=300,
    learning_rate=0.1,
    depth=8,
    l2_leaf_reg=3,
    bagging_temperature=0,
    random_strength=2,
    random_seed=42,
    thread_count=-1,
    verbose=False,
)


def load_raw():
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    return train, test


def base_xy(train):
    y = train["Will_Buy_EV"].map({"No": 0, "Yes": 1})
    X = train.drop(columns=["Will_Buy_EV"])
    return X, y


def holdout_split(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def cat_cols_of(X):
    return X.select_dtypes(include=["object", "str"]).columns.tolist()


def log_result(name, description, score, extra=None, mode="holdout"):
    row = {
        "name": name,
        "description": description,
        "mode": mode,
        "score": score,
        "extra": extra or {},
    }
    with open(RESULTS_PATH, "a") as f:
        f.write(json.dumps(row) + "\n")
    print(f"[{mode}] {name}: {score:.6f}  -- {description}")
    return row


def run_holdout(X, y, params=None, cat_features=None, early_stopping_rounds=50, use_eval_set=True):
    params = {**BASE_CATBOOST_PARAMS, **(params or {})}
    X_train, X_valid, y_train, y_valid = holdout_split(X, y)
    if cat_features is None:
        cat_features = cat_cols_of(X)
    model = CatBoostClassifier(**params)
    fit_kwargs = dict(cat_features=cat_features)
    if use_eval_set:
        fit_kwargs["eval_set"] = (X_valid, y_valid)
        fit_kwargs["early_stopping_rounds"] = early_stopping_rounds
    t0 = time.time()
    model.fit(X_train, y_train, **fit_kwargs)
    dt = time.time() - t0
    pred = model.predict_proba(X_valid)[:, 1]
    score = roc_auc_score(y_valid, pred)
    return score, model, dt


def run_kfold(X, y, params=None, cat_features=None, n_splits=5, early_stopping_rounds=50):
    params = {**BASE_CATBOOST_PARAMS, **(params or {})}
    if cat_features is None:
        cat_features = cat_cols_of(X)
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X, y), 1):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
        model = CatBoostClassifier(**params)
        model.fit(
            X_tr, y_tr,
            cat_features=cat_features,
            eval_set=(X_va, y_va),
            early_stopping_rounds=early_stopping_rounds,
        )
        pred = model.predict_proba(X_va)[:, 1]
        s = roc_auc_score(y_va, pred)
        scores.append(s)
        print(f"    fold {fold}: {s:.6f}")
    return np.mean(scores), np.std(scores), scores
