"""
Document discovery pipeline for RBA Statements on Monetary Policy (SMP) and
Board minutes, 2007-2026.

Run this locally (needs internet access) — not inside this chat environment.
    pip install requests beautifulsoup4 pandas

What it does:
1. Crawls each year's index page for Board minutes and extracts every
   individual minutes URL for that year.
2. Builds the SMP URL list directly from the known publication pattern
   (published every February, May, August, November).
3. Saves a master index (CSV) of every document to collect: type, year,
   month, url. This is the input list for text_preprocessing.py.

URL patterns confirmed from the RBA site (as of 2026):
  Minutes index (per year): https://www.rba.gov.au/monetary-policy/rba-board-minutes/{year}/
  Minutes document:         https://www.rba.gov.au/monetary-policy/rba-board-minutes/{year}/{year}-{mm}-{dd}.html
  SMP index (per year):     https://www.rba.gov.au/publications/smp/{year}/{month}/
  SMP document (PDF):       https://www.rba.gov.au/publications/smp/{year}/{month}/pdf/statement-on-monetary-policy-{year}-{mm}.pdf

Note: minutes are published under the "Reserve Bank Board" for 2007-2025
(~11 meetings/year, dates vary) and under the "Monetary Policy Board" from
March 2025 onward (8 meetings/year, dates announced in the schedule). The
minutes index page for each year lists the actual dates — that's why we
crawl it instead of guessing dates programmatically.
"""

import re
import time
import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import date

BASE = "https://www.rba.gov.au"
START_YEAR = 2007
END_YEAR = 2026

SMP_MONTHS = [("feb", "02"), ("may", "05"), ("aug", "08"), ("nov", "11")]


def discover_minutes_urls(year: int) -> list[dict]:
    """Fetch a year's minutes index page and extract every minutes document URL.

    The RBA has used two URL formats over time:
      - New (roughly 2015 onward): {year}-{mm}-{dd}.html   e.g. 2026-05-05.html
      - Old (roughly 2007-2014):    {dd}{mm}{year}.html     e.g. 06072010.html
    Both are matched here.
    """
    index_url = f"{BASE}/monetary-policy/rba-board-minutes/{year}/"
    resp = requests.get(index_url, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.content, "html.parser")

    pattern_new = re.compile(rf"/monetary-policy/rba-board-minutes/{year}/{year}-\d{{2}}-\d{{2}}\.html")
    pattern_old = re.compile(rf"/monetary-policy/rba-board-minutes/{year}/\d{{2}}\d{{2}}{year}\.html")

    urls = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if pattern_new.search(href) or pattern_old.search(href):
            full_url = href if href.startswith("http") else BASE + href
            urls.add(full_url)

    return [{"type": "minutes", "year": year, "url": u} for u in sorted(urls)]


def build_smp_urls(year: int) -> list[dict]:
    """SMP is published on a fixed quarterly schedule — build URLs directly,
    skipping quarters that haven't been published yet."""
    today = date.today()
    rows = []
    for month_name, month_num in SMP_MONTHS:
        # Skip if this quarter's publication month hasn't arrived yet
        if year > today.year or (year == today.year and int(month_num) > today.month):
            continue
        page_url = f"{BASE}/publications/smp/{year}/{month_name}/"
        pdf_url = f"{BASE}/publications/smp/{year}/{month_name}/pdf/statement-on-monetary-policy-{year}-{month_num}.pdf"
        rows.append({"type": "smp", "year": year, "month": month_name, "page_url": page_url, "pdf_url": pdf_url})
    return rows




def build_full_index() -> pd.DataFrame:
    all_rows = []

    print("Discovering Board minutes URLs...")
    for year in range(START_YEAR, END_YEAR + 1):
        try:
            rows = discover_minutes_urls(year)
            print(f"  {year}: {len(rows)} minutes found")
            all_rows.extend(rows)
        except requests.RequestException as e:
            print(f"  {year}: failed ({e})")
        time.sleep(0.5)  # be polite to the server

    minutes_df = pd.DataFrame(all_rows)

    print("\nBuilding SMP URL list (fixed quarterly schedule)...")
    smp_rows = []
    for year in range(START_YEAR, END_YEAR + 1):
        smp_rows.extend(build_smp_urls(year))
    smp_df = pd.DataFrame(smp_rows)

    print(f"\nTotal minutes: {len(minutes_df)}")
    print(f"Total SMP (before filtering future/unpublished quarters): {len(smp_df)}")

    minutes_df.to_csv("minutes_index.csv", index=False)
    smp_df.to_csv("smp_index.csv", index=False)
    print("\nSaved minutes_index.csv and smp_index.csv")

    return minutes_df, smp_df


if __name__ == "__main__":
    minutes_df, smp_df = build_full_index()
    print("\nNext step: filter smp_index.csv to remove quarters not yet published,")
    print("then feed both CSVs into text_preprocessing.py to fetch, clean and chunk")
    print("every document.")
