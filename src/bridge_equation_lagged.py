"""
Bridge equation — VARIANT with lagged cash rate.

This is a separate script from bridge_equation.py, so the original
baseline results are never overwritten. Use this to compare whether a
lagged cash rate (capturing monetary policy transmission lags) performs
better than the contemporaneous version.

Specification:
    cpi_yoy(t) = b0 + b1 * cpi_yoy(t-1) + b2 * cash_rate(t-4) + error

cash_rate(t-4) = the cash rate from 4 quarters (1 year) ago — a common
transmission-lag assumption in monetary policy literature.

Requirements:
    pip install pandas statsmodels scikit-learn
"""

import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

MACRO_PATH = "../data/processed/macro_clean.xlsx"
TARGET_COL = "cpi_pct_change_y_rba"
CASH_RATE_COL = "cash_rate"
QUARTER_COL = "quarter"

CASH_RATE_LAG = 4  # quarters — change to 2 to test a 6-month lag instead
TEST_SIZE_FRACTION = 0.2


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(QUARTER_COL).reset_index(drop=True)
    df["cpi_yoy_lag1"] = df[TARGET_COL].shift(1)
    df["cash_rate_lag"] = df[CASH_RATE_COL].shift(CASH_RATE_LAG)
    df = df.dropna(subset=["cpi_yoy_lag1", "cash_rate_lag", TARGET_COL]).reset_index(drop=True)
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
    print(f"  {len(df)} usable quarters after lagging (lost {CASH_RATE_LAG} extra quarters vs. original due to cash rate lag)")

    train, test = chronological_split(df, TEST_SIZE_FRACTION)
    print(f"  Train: {len(train)} quarters ({train[QUARTER_COL].iloc[0]} to {train[QUARTER_COL].iloc[-1]})")
    print(f"  Test:  {len(test)} quarters ({test[QUARTER_COL].iloc[0]} to {test[QUARTER_COL].iloc[-1]})")

    X_train = sm.add_constant(train[["cpi_yoy_lag1", "cash_rate_lag"]])
    y_train = train[TARGET_COL]

    print(f"\nFitting Bridge equation with cash_rate lagged {CASH_RATE_LAG} quarters...")
    model = sm.OLS(y_train, X_train).fit()
    print(model.summary())

    X_test = sm.add_constant(test[["cpi_yoy_lag1", "cash_rate_lag"]], has_constant="add")
    y_test = test[TARGET_COL]
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    print(f"\n=== Bridge Equation (cash_rate lag={CASH_RATE_LAG}) — Out-of-sample performance ===")
    print(f"RMSE: {rmse:.3f}")
    print(f"MAE:  {mae:.3f}")
    print("\nCompare these against the original (contemporaneous cash rate): RMSE 0.629, MAE 0.557")

    results = test[[QUARTER_COL, TARGET_COL]].copy()
    results["predicted"] = y_pred.values
    results.to_csv("../data/processed/bridge_equation_lagged_predictions.csv", index=False)
    print("\nSaved predictions to ../data/processed/bridge_equation_lagged_predictions.csv")
    print("(Original file bridge_equation_predictions.csv was NOT touched.)")


if __name__ == "__main__":
    main()