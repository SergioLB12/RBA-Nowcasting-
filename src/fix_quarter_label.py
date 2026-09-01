import pandas as pd

macro = pd.read_excel("../data/processed/macro_clean.xlsx")
macro["quarter"] = macro["quarter"].replace("2017Q4", "2017-Q4")
macro.to_excel("../data/processed/macro_clean.xlsx", index=False)
print("Fixed.")