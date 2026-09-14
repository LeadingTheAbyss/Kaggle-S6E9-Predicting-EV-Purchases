"""Reproduce the exact holdout score already logged in eda.ipynb (0.9410766) as a sanity check
that this harness (same split, same params) is faithful before trusting any deltas from it."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_raw, base_xy, run_holdout, log_result

train, test = load_raw()
X, y = base_xy(train)

score, model, dt = run_holdout(X, y, early_stopping_rounds=None, use_eval_set=False)
log_result("baseline_raw", "Exact repro of notebook's tuned CatBoost holdout run (no early stopping, 300 iters)", score, {"fit_seconds": dt})

score_es, model_es, dt_es = run_holdout(X, y)
log_result("baseline_early_stop", "Same as baseline but with early_stopping_rounds=50 on holdout eval_set (used for all later screening runs for speed)", score_es, {"fit_seconds": dt_es, "best_iteration": model_es.get_best_iteration()})
