import pandas as pd

final = pd.read_csv("../data/processed/quarterly_features.csv")
missing = final[final["n_documents"].isna()]

print(f"Quarters with macro data but no text match: {len(missing)}")
print(missing[["quarter"]])