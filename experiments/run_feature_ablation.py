"""
Screen each feature-engineering candidate individually against the base feature set,
on the fixed holdout (fast, early-stopped). Each is additive-only (base features + one
new idea) so any delta is attributable to that idea alone, not to removing something else.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_raw, base_xy, run_holdout, log_result, cat_cols_of
from features import FEATURE_FUNCS

train, _ = load_raw()
X_base, y = base_xy(train)

for name, fn in FEATURE_FUNCS.items():
    X_aug = fn(X_base)
    score, model, dt = run_holdout(X_aug, y)
    log_result(
        f"feature__{name}",
        fn.__doc__.strip().split("\n")[0] if fn.__doc__ else name,
        score,
        {"fit_seconds": dt, "n_features": X_aug.shape[1], "best_iteration": model.get_best_iteration()},
    )
