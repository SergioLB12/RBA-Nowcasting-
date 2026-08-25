"""
Starter script to build the macro dataset for the capstone.
Run this locally (needs internet access) — not inside this chat environment.

Requirements:
    pip install readrba pandas requests

Note: readrba is an R package. For a pure-Python alternative, this script
uses direct RBA xlsx downloads + the ABS Data API (SDMX).
"""

import pandas as pd
import requests
from io import BytesIO

# ---------------------------------------------------------------
# 1. RBA Table G1 — Consumer Price Inflation (quarterly, re-referenced)
# ---------------------------------------------------------------
RBA_G1_URL = "https://www.rba.gov.au/statistics/tables/xls/g01hist.xlsx"
resp = requests.get(RBA_G1_URL)
cpi_raw = pd.read_excel(BytesIO(resp.content), sheet_name="Data", skiprows=10)
cpi_raw.to_csv("rba_g1_cpi_raw.csv", index=False)
print("Saved rba_g1_cpi_raw.csv — inspect and select the headline CPI column.")

# ---------------------------------------------------------------
# 2. RBA Table F1 — Cash rate
# ---------------------------------------------------------------
RBA_F1_URL = "https://www.rba.gov.au/statistics/tables/xls/f01hist.xlsx"
resp = requests.get(RBA_F1_URL)
cash_raw = pd.read_excel(BytesIO(resp.content), sheet_name="Data", skiprows=10)
cash_raw.to_csv("rba_f1_cashrate_raw.csv", index=False)
print("Saved rba_f1_cashrate_raw.csv")

# ---------------------------------------------------------------
# 3. ABS Labour Force (via Data API, SDMX-JSON) — placeholder query
#    Docs: https://www.abs.gov.au/about/data-services/application-programming-interface-api
#    You'll need to find the exact dataflow/series key for Labour Force (6202.0)
#    on the ABS Data API Explorer before this call works as-is.
# ---------------------------------------------------------------
# ABS_API_URL = "https://api.data.abs.gov.au/data/{dataflow}/{series_key}?startPeriod=2005&format=csv"
# labour = pd.read_csv(ABS_API_URL)
# labour.to_csv("abs_labour_force_raw.csv", index=False)

print("\nNext step: open the RBA CSVs, confirm the correct row/column for the")
print("headline CPI and cash rate series, then merge on quarter-end date.")
