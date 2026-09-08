"""
Random Forest — macro variables only (Model 2 of 3).

Uses the SAME two predictors as the Bridge equation baseline
(cpi_yoy_lag1, cash_rate), so the comparison between classical
econometrics and ML is a fair, apples-to-apples test of the model class
itself — not confounded by different inputs.

Hyperparameters are deliberately conservative given the small sample
(~77 observations): a shallow, small forest reduces overfitting risk,
consistent with the literature finding (RBA, 2025) that simple/sparse
models outperform complex ones at this sample size.

Requirements:
    pip install pandas scikit-learn
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

MACRO_PATH = "../data/processed/macro_clean.xlsx"
TARGET_COL = "cpi_pct_change_y_rba"
CASH_RATE_COL = "cash_rate"
QUARTER_COL = "quarter"

TEST_SIZE_FRACTION = 0.2
RANDOM_STATE = 42


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(QUARTER_COL).reset_index(drop=True)
    df["cpi_yoy_lag1"] = df[TARGET_COL].shift(1)
    df = df.dropna(subset=["cpi_yoy_lag1", TARGET_COL, CASH_RATE_COL]).reset_index(drop=True)
    return df


def chronological_split(df: pd.DataFrame, test_fraction: float):
    n_test = max(1, int(len(df) * test_fraction))
    train = df.iloc[: -n_test]
    test = df.iloc[-n_test:]
    return train, test


def main():
    print("Loading macro dataset...")
    df = pd.read_excel(MACRO_PATH)
    df = build_features(df)
    print(f"  {len(df)} usable quarters after lagging")

    train, test = chronological_split(df, TEST_SIZE_FRACTION)
    print(f"  Train: {len(train)} quarters ({train[QUARTER_COL].iloc[0]} to {train[QUARTER_COL].iloc[-1]})")
    print(f"  Test:  {len(test)} quarters ({test[QUARTER_COL].iloc[0]} to {test[QUARTER_COL].iloc[-1]})")

    feature_cols = ["cpi_yoy_lag1", CASH_RATE_COL]
    X_train, y_train = train[feature_cols], train[TARGET_COL]
    X_test, y_test = test[feature_cols], test[TARGET_COL]

    print("\nFitting Random Forest (macro only)...")
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=3,          # shallow trees — small sample, avoid overfitting
        min_samples_leaf=3,   # each leaf needs at least 3 quarters of support
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    print("\n=== Random Forest (macro only) — Out-of-sample performance ===")
    print(f"RMSE: {rmse:.3f}")
    print(f"MAE:  {mae:.3f}")
    print("\nCompare against Bridge equation baseline: RMSE 0.629, MAE 0.557")

    print("\n=== Feature importances ===")
    for name, importance in zip(feature_cols, model.feature_importances_):
        print(f"  {name}: {importance:.3f}")

    results = test[[QUARTER_COL, TARGET_COL]].copy()
    results["predicted"] = y_pred
    results.to_csv("../data/processed/random_forest_macro_predictions.csv", index=False)
    print("\nSaved predictions to ../data/processed/random_forest_macro_predictions.csv")


if __name__ == "__main__":
    main()