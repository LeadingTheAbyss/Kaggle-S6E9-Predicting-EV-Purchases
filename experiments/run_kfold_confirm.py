"""
Confirm a candidate feature set with full 5-fold CV (robust estimate) rather than a
single holdout, since a holdout can move +/-0.001 from fold noise alone (the notebook's
own 5-fold std was ~0.00085). Usage:

    python run_kfold_confirm.py base
    python run_kfold_confirm.py all_candidates
    python run_kfold_confirm.py charging_infra income_x_subsidy   # combine specific winners
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_raw, base_xy, run_kfold, log_result
from features import FEATURE_FUNCS

names = sys.argv[1:] or ["base"]
train, _ = load_raw()
X, y = base_xy(train)

if names != ["base"]:
    for n in names:
        X = FEATURE_FUNCS[n](X)

mean_s, std_s, scores = run_kfold(X, y)
label = "+".join(names)
log_result(f"kfold__{label}", f"5-fold CV confirmation of feature set: {label}", mean_s, {"std": std_s, "fold_scores": scores, "n_features": X.shape[1]}, mode="kfold")
