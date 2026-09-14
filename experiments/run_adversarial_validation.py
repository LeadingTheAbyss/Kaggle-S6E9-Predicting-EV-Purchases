"""
Validation experiment: adversarial validation.

Hypothesis to test: if train and test come from the same generating process (typical
for Kaggle Playground synthetic series), a classifier trying to distinguish
"is this row from train or test" should score close to 0.5 AUC. If it scores much
higher, there's train/test drift -- which would explain any gap between local CV and
the public leaderboard, and would mean holdout CV is an unreliable proxy for LB score.

This directly checks whether the 0.9411 (local CV) vs 0.94083 (Kaggle) gap is
distribution drift or just ordinary CV variance (std was ~0.0009 across folds).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier
from common import load_raw, log_result

train, test = load_raw()
train_feat = train.drop(columns=["Will_Buy_EV"])
test_feat = test.copy()

train_feat["__is_test"] = 0
test_feat["__is_test"] = 1
combined = pd.concat([train_feat, test_feat], axis=0, ignore_index=True)

y = combined["__is_test"]
X = combined.drop(columns=["__is_test"])
cat_features = X.select_dtypes(include=["object", "str"]).columns.tolist()

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = []
importances = None
for tr_idx, va_idx in kf.split(X, y):
    model = CatBoostClassifier(
        iterations=200, depth=6, learning_rate=0.1, random_seed=42,
        verbose=False, thread_count=-1,
    )
    model.fit(X.iloc[tr_idx], y.iloc[tr_idx], cat_features=cat_features)
    pred = model.predict_proba(X.iloc[va_idx])[:, 1]
    scores.append(roc_auc_score(y.iloc[va_idx], pred))
    fi = model.get_feature_importance()
    importances = fi if importances is None else importances + fi

mean_score = sum(scores) / len(scores)
importances = importances / len(scores)
top_feats = sorted(zip(X.columns, importances), key=lambda t: -t[1])[:5]

log_result(
    "adversarial_validation",
    "train-vs-test classifier AUC; ~0.5 means no distribution drift, holdout CV should track LB",
    mean_score,
    {"fold_scores": scores, "top_distinguishing_features": top_feats},
    mode="diagnostic",
)
