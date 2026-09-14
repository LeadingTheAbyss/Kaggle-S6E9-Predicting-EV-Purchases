"""
Feature engineering candidates, each as a standalone function that takes the raw
train/test-style dataframe (post id-drop, pre-split) and returns an augmented copy.

Every idea here follows the same shape: a hypothesis about WHY it should help,
tested empirically rather than assumed. Trees (CatBoost) already learn arbitrary
splits/interactions given enough depth, so most "interaction feature" engineering
in GBMs pays off only when it either (a) exposes a pattern that needs more depth than
the model is currently using to reconstruct on its own, or (b) reduces noise by
collapsing a ratio/rate the model would otherwise have to approximate via many splits.
"""
import numpy as np
import pandas as pd


def add_income_x_subsidy(df):
    """
    Hypothesis: subsidy effect is not additive with income -- e.g. subsidy may swing
    a middle-income buyer far more than a already-affluent one who'd buy anyway, or
    vice versa. Explicit interaction saves the tree from needing depth >= 2 splits
    (subsidy, then income) in a single path just to reconstruct this.
    """
    out = df.copy()
    out["Subsidy_Flag"] = (out["Subsidy_Available"] == "Yes").astype(int)
    out["Income_x_Subsidy"] = out["Annual_Income_USD"] * out["Subsidy_Flag"]
    return out


def add_charging_infra_features(df):
    """
    Hypothesis (from README next-steps): raw station counts near home/work matter less
    than total accessible infrastructure, and infrastructure relative to how much you
    actually drive (commute) is what should predict range-anxiety-driven purchase decisions.
    """
    out = df.copy()
    out["Total_Charging_Stations"] = out["Charging_Stations_Near_Home"] + out["Charging_Stations_Near_Work"]
    out["Charging_Station_Ratio_Commute"] = out["Total_Charging_Stations"] / (out["Daily_Commute_km"] + 1.0)
    return out


def add_affordability_features(df):
    """
    Hypothesis: per-person affordability (income spread across cars already owned as a proxy
    for household size / existing vehicle spend) is a better wealth signal than raw income,
    and commute burden relative to income may separate "needs a cheap efficient car" buyers
    from "commute is irrelevant, income dominates" buyers.
    """
    out = df.copy()
    out["Income_per_Car"] = out["Annual_Income_USD"] / out["Number_of_Cars_Owned"]
    out["Commute_to_Income"] = out["Daily_Commute_km"] / (out["Annual_Income_USD"] / 1000.0)
    return out


def add_ordinal_encodings(df):
    """
    Hypothesis: Range_Anxiety_Level (Low/Medium/High) and Environmental_Concern_Level
    (already 1-5 numeric) carry ordinal information that CatBoost's categorical target
    stats partially discard by treating Range_Anxiety_Level as unordered. Giving it as
    an explicit ordinal integer lets splits use "<=" thresholds instead of set-membership.
    """
    out = df.copy()
    order = {"Low": 0, "Medium": 1, "High": 2}
    out["Range_Anxiety_Ordinal"] = out["Range_Anxiety_Level"].map(order)
    return out


def add_concern_anxiety_interaction(df):
    """
    Hypothesis: high environmental concern but high range anxiety may cancel out (wants an
    EV, afraid of running out), while high concern + low anxiety should be the strongest
    positive segment. A single combined feature exposes this cancellation directly.
    """
    out = df.copy()
    anxiety_num = out["Range_Anxiety_Level"].map({"Low": 0, "Medium": 1, "High": 2})
    out["Concern_minus_Anxiety"] = out["Environmental_Concern_Level"] - anxiety_num
    return out


def add_city_charging_interaction(df):
    """
    Hypothesis: home-charging feasibility means different things in different city types
    (a "Yes" in Urban likely means a rarer dedicated parking spot vs. easy in Rural/Suburban).
    Combining City_Type + Home_Charging_Possible into one categorical lets CatBoost's
    per-category target stats reflect that context directly instead of needing both features
    split on in the same path.
    """
    out = df.copy()
    out["City_x_HomeCharging"] = out["City_Type"].astype(str) + "_" + out["Home_Charging_Possible"].astype(str)
    return out


def add_all_candidates(df):
    out = df
    out = add_income_x_subsidy(out)
    out = add_charging_infra_features(out)
    out = add_affordability_features(out)
    out = add_ordinal_encodings(out)
    out = add_concern_anxiety_interaction(out)
    out = add_city_charging_interaction(out)
    return out


FEATURE_FUNCS = {
    "income_x_subsidy": add_income_x_subsidy,
    "charging_infra": add_charging_infra_features,
    "affordability": add_affordability_features,
    "ordinal_encodings": add_ordinal_encodings,
    "concern_anxiety_interaction": add_concern_anxiety_interaction,
    "city_charging_interaction": add_city_charging_interaction,
    "all_candidates": add_all_candidates,
}
