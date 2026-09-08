"""
Random Forest + Text (Model 3 of 3) — the model that directly answers the
research question: does adding RBA text improve on macro-only nowcasting?

Same macro predictors as Models 1 and 2 (cpi_yoy_lag1, cash_rate), PLUS two
types of text-derived features:
  - sentiment: Loughran-McDonald dictionary score (already 1 number, used as-is)
  - embeddings: Sentence-BERT vectors (384 dimensions) — reduced to a small
    number of principal components before modeling. With only ~62 training
    quarters, using all 384 embedding dimensions directly would guarantee
    overfitting (more features than observations). PCA is fit ONLY on the
    training set and then applied to test, to avoid any information leakage
    from the test period into the dimensionality reduction step.

Requirements:
    pip install pandas scikit-learn numpy
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, mean_absolute_error

FEATURES_PATH = "../data/processed/quarterly_features.csv"
TARGET_COL = "cpi_pct_change_y_rba"
CASH_RATE_COL = "cash_rate"
QUARTER_COL = "quarter"

N_PCA_COMPONENTS = 2   # keep low given the small sample — see note above
TEST_SIZE_FRACTION = 0.2
RANDOM_STATE = 42


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(QUARTER_COL).reset_index(drop=True)
    df["cpi_yoy_lag1"] = df[TARGET_COL].shift(1)
    df = df.dropna(subset=["cpi_yoy_lag1", TARGET_COL, CASH_RATE_COL, "sentiment"]).reset_index(drop=True)
    return df


def chronological_split(df: pd.DataFrame, test_fraction: float):
    n_test = max(1, int(len(df) * test_fraction))
    train = df.iloc[: -n_test]
    test = df.iloc[-n_test:]
    return train, test


def main():
    print("Loading merged macro + text features dataset...")
    df = pd.read_csv(FEATURES_PATH)
    df = build_features(df)
    print(f"  {len(df)} usable quarters after lagging")

    embedding_cols = [c for c in df.columns if c.startswith("emb_")]
    print(f"  Found {len(embedding_cols)} embedding dimensions -> reducing to {N_PCA_COMPONENTS} components via PCA")

    train, test = chronological_split(df, TEST_SIZE_FRACTION)
    print(f"  Train: {len(train)} quarters ({train[QUARTER_COL].iloc[0]} to {train[QUARTER_COL].iloc[-1]})")
    print(f"  Test:  {len(test)} quarters ({test[QUARTER_COL].iloc[0]} to {test[QUARTER_COL].iloc[-1]})")

    # Fit PCA on TRAIN embeddings only, then apply to both train and test —
    # this prevents test-period information from leaking into the reduction.
    pca = PCA(n_components=N_PCA_COMPONENTS, random_state=RANDOM_STATE)
    train_emb_pca = pca.fit_transform(train[embedding_cols])
    test_emb_pca = pca.transform(test[embedding_cols])
    print(f"  PCA explained variance ratio: {pca.explained_variance_ratio_.round(3).tolist()}")

    pca_cols = [f"emb_pc{i+1}" for i in range(N_PCA_COMPONENTS)]
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    train[pca_cols] = train_emb_pca
    test[pca_cols] = test_emb_pca

    feature_cols = ["cpi_yoy_lag1", CASH_RATE_COL, "sentiment"] + pca_cols
    X_train, y_train = train[feature_cols], train[TARGET_COL]
    X_test, y_test = test[feature_cols], test[TARGET_COL]

    print(f"\nFitting Random Forest + Text ({len(feature_cols)} features: {feature_cols})...")
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=3,
        min_samples_leaf=3,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    print("\n=== Random Forest + Text — Out-of-sample performance ===")
    print(f"RMSE: {rmse:.3f}")
    print(f"MAE:  {mae:.3f}")
    print("\nCompare against:")
    print("  Bridge equation (baseline):    RMSE 0.629, MAE 0.557")
    print("  Random Forest (macro only):    RMSE 1.232, MAE 0.911")

    print("\n=== Feature importances ===")
    for name, importance in zip(feature_cols, model.feature_importances_):
        print(f"  {name}: {importance:.3f}")

    results = test[[QUARTER_COL, TARGET_COL]].copy()
    results["predicted"] = y_pred
    results.to_csv("../data/processed/random_forest_text_predictions.csv", index=False)
    print("\nSaved predictions to ../data/processed/random_forest_text_predictions.csv")


if __name__ == "__main__":
    main()