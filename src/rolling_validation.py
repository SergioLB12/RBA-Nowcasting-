"""
 Rolling out-of-sample validation across all three specifications.

Instead of a single 80/20 chronological split, this expands the training
window one quarter at a time: train on all data up to quarter (t-1),
predict quarter t, then move forward and repeat. This produces many
out-of-sample predictions instead of one static test set — a much more
robust check of how each model generalizes over time, and whether the
single-holdout results (Bridge 0.629, RF-macro 1.232, RF+Text 1.292)
hold up or were partly an artifact of that particular test window.

For Random Forest + Text, PCA is refit at EVERY window using only that
window's training data (never the point being predicted) — the same
no-leakage principle from the original random_forest_text.py, just
repeated at each step.

Requirements:
    pip install pandas scikit-learn statsmodels numpy
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, mean_absolute_error

FEATURES_PATH = "../data/processed/quarterly_features.csv"
TARGET_COL = "cpi_pct_change_y_rba"
CASH_RATE_COL = "cash_rate"
QUARTER_COL = "quarter"

MIN_TRAIN_SIZE = 40   # first window trains on the oldest 40 quarters
N_PCA_COMPONENTS = 2
RANDOM_STATE = 42


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(QUARTER_COL).reset_index(drop=True)
    df["cpi_yoy_lag1"] = df[TARGET_COL].shift(1)
    df = df.dropna(subset=["cpi_yoy_lag1", TARGET_COL, CASH_RATE_COL, "sentiment"]).reset_index(drop=True)
    return df


def predict_bridge(train, test_row):
    X_train = sm.add_constant(train[["cpi_yoy_lag1", CASH_RATE_COL]])
    y_train = train[TARGET_COL]
    model = sm.OLS(y_train, X_train).fit()
    X_test = sm.add_constant(test_row[["cpi_yoy_lag1", CASH_RATE_COL]], has_constant="add")
    return model.predict(X_test).iloc[0]


def predict_rf_macro(train, test_row):
    feature_cols = ["cpi_yoy_lag1", CASH_RATE_COL]
    model = RandomForestRegressor(n_estimators=200, max_depth=3, min_samples_leaf=3, random_state=RANDOM_STATE)
    model.fit(train[feature_cols], train[TARGET_COL])
    return model.predict(test_row[feature_cols])[0]


def predict_rf_text(train, test_row, embedding_cols):
    pca = PCA(n_components=N_PCA_COMPONENTS, random_state=RANDOM_STATE)
    train_emb_pca = pca.fit_transform(train[embedding_cols])
    test_emb_pca = pca.transform(test_row[embedding_cols])

    pca_cols = [f"emb_pc{i+1}" for i in range(N_PCA_COMPONENTS)]
    train = train.reset_index(drop=True).copy()
    train[pca_cols] = train_emb_pca
    test_row = test_row.reset_index(drop=True).copy()
    test_row[pca_cols] = test_emb_pca

    feature_cols = ["cpi_yoy_lag1", CASH_RATE_COL, "sentiment"] + pca_cols
    model = RandomForestRegressor(n_estimators=200, max_depth=3, min_samples_leaf=3, random_state=RANDOM_STATE)
    model.fit(train[feature_cols], train[TARGET_COL])
    return model.predict(test_row[feature_cols])[0]


def main():
    print("Loading merged macro + text features dataset...")
    df = pd.read_csv(FEATURES_PATH)
    df = build_features(df)
    embedding_cols = [c for c in df.columns if c.startswith("emb_")]
    n_windows = len(df) - MIN_TRAIN_SIZE
    print(f"  {len(df)} usable quarters, {n_windows} rolling windows (first train size = {MIN_TRAIN_SIZE})")

    results = {"quarter": [], "actual": [], "bridge": [], "rf_macro": [], "rf_text": []}

    print("\nRunning rolling validation (this retrains 3 models per step, may take a minute)...")
    for i in range(MIN_TRAIN_SIZE, len(df)):
        train = df.iloc[:i]
        test_row = df.iloc[[i]]

        results["quarter"].append(test_row[QUARTER_COL].iloc[0])
        results["actual"].append(test_row[TARGET_COL].iloc[0])
        results["bridge"].append(predict_bridge(train, test_row))
        results["rf_macro"].append(predict_rf_macro(train, test_row))
        results["rf_text"].append(predict_rf_text(train, test_row, embedding_cols))

        if (i - MIN_TRAIN_SIZE + 1) % 10 == 0:
            print(f"  ...{i - MIN_TRAIN_SIZE + 1}/{n_windows} windows done")

    results_df = pd.DataFrame(results)
    results_df.to_csv("../data/processed/rolling_validation_predictions.csv", index=False)

    print(f"\n=== Rolling out-of-sample performance ({n_windows} windows, {results_df['quarter'].iloc[0]} to {results_df['quarter'].iloc[-1]}) ===\n")
    print(f"{'Model':<20}{'RMSE':>10}{'MAE':>10}")
    for model_name, col in [("Bridge equation", "bridge"), ("Random Forest (macro)", "rf_macro"), ("Random Forest+Text", "rf_text")]:
        rmse = np.sqrt(mean_squared_error(results_df["actual"], results_df[col]))
        mae = mean_absolute_error(results_df["actual"], results_df[col])
        print(f"{model_name:<20}{rmse:>10.3f}{mae:>10.3f}")

    print("\nCompare against the original single-holdout results:")
    print("  Bridge equation:        RMSE 0.629, MAE 0.557")
    print("  Random Forest (macro):  RMSE 1.232, MAE 0.911")
    print("  Random Forest + Text:   RMSE 1.292, MAE 0.933")
    print("\nSaved full predictions to ../data/processed/rolling_validation_predictions.csv")


if __name__ == "__main__":
    main()