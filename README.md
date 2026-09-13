<div align="center">

# Predicting Electric Vehicle Purchases

Modeling consumer intent to buy an EV from tabular behavioral data.

[![Competition](https://img.shields.io/badge/Kaggle-Playground_S6E9-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/competitions/playground-series-s6e9)
[![Metric](https://img.shields.io/badge/Metric-ROC--AUC-success)]()
[![Notebook](https://img.shields.io/badge/Notebook-eda.ipynb-F37626?logo=jupyter&logoColor=white)](./eda.ipynb)

<img src="./public/img1.png" alt="Competition screenshot" width="700">

<a href="https://www.kaggle.com/competitions/playground-series-s6e9"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open in Kaggle"></a>

[Competition](https://www.kaggle.com/competitions/playground-series-s6e9) · [Kaggle Profile](https://www.kaggle.com/Masochistic) · [GitHub](https://github.com/LeadingTheAbyss)

</div>

This repo has the EDA and modeling pipeline for Playground Series S6E9: Predicting Electric Vehicle Purchases. Binary classification: given demographic, financial, and infrastructure features about a person, predict whether they buy an EV (`Will_Buy_EV`).

The notebook covers baselining, feature scaling, model comparison, and the EDA that found the two strongest signals in the data, before landing on a CatBoost model trained on raw categorical features.

## Problem setting

Each row is one person:

- demographics: `Age`, `Gender`, `Annual_Income_USD`
- usage patterns: `Daily_Commute_km`, `Number_of_Cars_Owned`, `Current_Car_Type`
- infrastructure access: `Charging_Stations_Near_Home`, `Charging_Stations_Near_Work`, `Home_Charging_Possible`
- context & incentives: `City_Type`, `Subsidy_Available`, `Environmental_Concern_Level`, `Range_Anxiety_Level`

Target `Will_Buy_EV` is binary (`Yes` / `No`), scored on ROC-AUC.

> [!NOTE]
> Train: 668,665 rows. Test: 286,571 rows. No missing values, no leaked identifiers beyond `id`.

## Baseline: logistic regression

One-hot encode categoricals with `pd.get_dummies`, fit a plain logistic regression: `P(Will_Buy_EV=1 | X) = sigmoid(w·X + b)`.

| Step | ROC-AUC |
|---|---|
| One-hot encoding only | 0.9049 |
| + `StandardScaler` | 0.9384 |

**Why scaling mattered:** logistic regression converges faster and finds a better optimum when features share a comparable scale. `Annual_Income_USD` (range ~30K-190K) was dominating the loss landscape next to binary dummy columns. Standardizing alone added +0.033.

## Moving to gradient boosting

Tree-based models don't need scaling and pick up non-linear interactions (income × subsidy availability) that a linear model misses.

| Model | Config | ROC-AUC |
|---|---|---|
| XGBoost | `n_estimators=300, depth=6, lr=0.05` | 0.9417 |
| CatBoost | `iterations=300, depth=6, lr=0.05` | 0.9409 |

CatBoost trailed XGBoost by 0.0008 here, but this is on one-hot encoded data, before using CatBoost's actual advantage: native categorical handling.

## EDA: finding the signal

### Income is strongly predictive

```
Annual_Income_USD (quintile)     P(Will_Buy_EV = Yes)
(30K, 62.7K]                     0.065
(62.7K, 78.9K]                   0.119
(78.9K, 91.6K]                   0.168
(91.6K, 107.8K]                  0.209
(107.8K, 188.5K]                 0.313
```

Purchase likelihood rises monotonically with income, nearly 5x from bottom to top quintile.

### Subsidy availability is the strongest single feature

```
Subsidy_Available     P(Will_Buy_EV = Yes)
No                     0.006
Yes                    0.275
```

> [!TIP]
> No subsidy, almost nobody buys. With a subsidy, purchase probability jumps ~46x. This one categorical flag carries more separating power than any continuous feature in the dataset.

## Final model: CatBoost on native categoricals

Instead of one-hot encoding, categorical columns (`Gender`, `City_Type`, `Current_Car_Type`, `Home_Charging_Possible`, `Subsidy_Available`, `Range_Anxiety_Level`) go straight into CatBoost via `cat_features`. CatBoost's ordered target statistics handle both high- and low-cardinality categoricals without the dimensionality blow-up of one-hot encoding, and without the leakage naive mean-encoding introduces.

```python
model = CatBoostClassifier(
    iterations = 500,
    learning_rate = 0.05,
    depth = 6,
    random_seed = 42
)
model.fit(X_train, y_train, cat_features = cat_cols)
```

| Pipeline | ROC-AUC |
|---|---|
| One-hot + CatBoost | 0.9409 |
| Native categoricals + CatBoost (final) | 0.9411 |

Public leaderboard top score is 0.946, so this baseline sits about 0.005 behind. Reasonable for a first proper submission.

## Submission

```python
test_predictions = model.predict_proba(test)[:, 1]
submission = pd.DataFrame({
    "id": test["id"],
    "Will_Buy_EV": test_predictions
})
submission.to_csv("submission.csv", index = False)
```

`submission.csv` has two columns: `id` and `Will_Buy_EV` (predicted probability, not a hard label). Matches the competition's expected format.

## Summary

- income and subsidy availability are the two dominant predictors
- scaling matters a lot for linear models, not at all for boosted trees
- CatBoost with native categorical handling narrowly beats XGBoost + one-hot encoding
- current baseline: 0.9411 ROC-AUC, ~0.005 off the public leaderboard top

## Closing remarks

> [!IMPORTANT]
> Careful reading of the data went further than reaching for a bigger model. Two crosstabs (`Annual_Income_USD`, `Subsidy_Available` vs. target) explained more of the score than switching between XGBoost and CatBoost ever did.

Next steps:

- interaction features between `Subsidy_Available` and `Annual_Income_USD`
- hyperparameter tuning (`depth`, `learning_rate`, `iterations`) via cross-validation
- feature engineering from `Charging_Stations_Near_Home` / `Near_Work` (total accessible stations, ratio to commute distance)
- ensemble/blend of CatBoost and XGBoost predictions

<div align="center">

[Kaggle](https://www.kaggle.com/Masochistic) · [GitHub](https://github.com/LeadingTheAbyss) · [Competition Page](https://www.kaggle.com/competitions/playground-series-s6e9)

[↑ Back to top](#predicting-electric-vehicle-purchases)

</div>
