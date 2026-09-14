"""
Alternative-model experiments on the base feature set (same holdout split as everything else):

1. XGBoost with native categorical support (enable_categorical=True) instead of one-hot --
   the notebook only tried XGBoost on one-hot data. XGBoost >= 1.5 can split on categoricals
   directly, closer to a fair fight against CatBoost's native handling.
2. CatBoost + XGBoost blend (simple probability average) -- hypothesis: the two models make
   partially uncorrelated errors (different categorical handling, different tree-building
   algorithm), so averaging should reduce variance even if neither is individually better.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from common import load_raw, base_xy, holdout_split, run_holdout, log_result, cat_cols_of, BASE_CATBOOST_PARAMS
from catboost import CatBoostClassifier

train, _ = load_raw()
X, y = base_xy(train)
X_train, X_valid, y_train, y_valid = holdout_split(X, y)
cat_features = cat_cols_of(X)

# --- XGBoost native categorical ---
X_train_xgb = X_train.copy()
X_valid_xgb = X_valid.copy()
for c in cat_features:
    X_train_xgb[c] = X_train_xgb[c].astype("category")
    X_valid_xgb[c] = X_valid_xgb[c].astype("category")

xgb_model = XGBClassifier(
    n_estimators=300, max_depth=8, learning_rate=0.1,
    enable_categorical=True, tree_method="hist",
    random_state=42, eval_metric="auc",
)
xgb_model.fit(X_train_xgb, y_train, eval_set=[(X_valid_xgb, y_valid)], verbose=False)
xgb_pred = xgb_model.predict_proba(X_valid_xgb)[:, 1]
xgb_score = roc_auc_score(y_valid, xgb_pred)
log_result("xgb_native_categorical", "XGBoost with enable_categorical=True (native cat splits, no one-hot)", xgb_score)

# --- CatBoost (same holdout, for blend) ---
cb_model = CatBoostClassifier(**{**BASE_CATBOOST_PARAMS, "verbose": False})
cb_model.fit(X_train, y_train, cat_features=cat_features)
cb_pred = cb_model.predict_proba(X_valid)[:, 1]
cb_score = roc_auc_score(y_valid, cb_pred)
log_result("catboost_same_holdout", "CatBoost tuned config on this holdout (for blend comparison)", cb_score)

# --- Blend ---
blend_pred = 0.5 * xgb_pred + 0.5 * cb_pred
blend_score = roc_auc_score(y_valid, blend_pred)
log_result("blend_catboost_xgb_50_50", "Simple 50/50 average of CatBoost + XGBoost native-categorical probabilities", blend_score)

# Weighted blend sweep
best_w, best_s = None, -1
for w in [0.3, 0.4, 0.5, 0.6, 0.7]:
    s = roc_auc_score(y_valid, w * cb_pred + (1 - w) * xgb_pred)
    if s > best_s:
        best_w, best_s = w, s
log_result("blend_best_weight", f"Best CatBoost-weight in blend sweep (w={best_w} for CatBoost)", best_s, {"weight_catboost": best_w})
