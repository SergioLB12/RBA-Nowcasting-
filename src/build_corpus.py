"""
Batch scraping pipeline: reads minutes_index.csv and smp_index.csv, fetches
each document, cleans it, and saves the result incrementally to a single
corpus CSV — so if it fails partway through, you don't lose progress.
"""

import os
import time
import pandas as pd
from text_preprocessing import fetch_clean_text, split_into_sections, document_stats
from text_preprocessing import MINUTES_SECTION_HEADERS, SMP_SECTION_HEADERS

OUTPUT_PATH = "../data/processed/text_corpus.csv"
MINUTES_INDEX = "../data/external/minutes_index.csv"
SMP_INDEX = "../data/external/smp_index.csv"


def load_existing_progress() -> tuple[pd.DataFrame, set]:
    """If the corpus file already exists (from a previous partial run), load
    it and figure out which document URLs are already done."""
    if os.path.exists(OUTPUT_PATH):
        existing = pd.read_csv(OUTPUT_PATH)
        done_urls = set(existing["source_url"].tolist())
        print(f"Found existing progress: {len(existing)} rows, {len(done_urls)} documents already done.")
        return existing, done_urls
    return pd.DataFrame(), set()


def process_document(doc_type: str, url: str, year, month=None) -> list[dict]:
    """Fetch, clean, and section-split one document. Returns one row per
    section (so a long SMP becomes several rows, one per section)."""
    headers = MINUTES_SECTION_HEADERS if doc_type == "minutes" else SMP_SECTION_HEADERS
    text = fetch_clean_text(url)
    sections = split_into_sections(text, headers)

    rows = []
    for section_name, section_text in sections.items():
        stats = document_stats(f"{doc_type}_{year}_{month or ''}", section_text)
        rows.append({
            "doc_type": doc_type,
            "year": year,
            "month": month,
            "section": section_name,
            "text": section_text,
            "n_words": stats["n_words"],
            "n_chars": stats["n_chars"],
            "source_url": url,
        })
    return rows


def build_corpus():
    existing_df, done_urls = load_existing_progress()
    all_rows = existing_df.to_dict("records") if not existing_df.empty else []

    minutes_df = pd.read_csv(MINUTES_INDEX)
    smp_df = pd.read_csv(SMP_INDEX)

    total = len(minutes_df) + len(smp_df)
    processed_count = len(done_urls)

    print(f"\nProcessing minutes ({len(minutes_df)} documents)...")
    for i, row in minutes_df.iterrows():
        if row["url"] in done_urls:
            continue
        try:
            rows = process_document("minutes", row["url"], row["year"])
            all_rows.extend(rows)
            processed_count += 1
            print(f"  [{processed_count}/{total}] OK: {row['url']}")
        except Exception as e:
            print(f"  [{processed_count}/{total}] FAILED: {row['url']} — {e}")
        if processed_count % 10 == 0:
            pd.DataFrame(all_rows).to_csv(OUTPUT_PATH, index=False)
        time.sleep(0.5)

    print(f"\nProcessing SMP ({len(smp_df)} documents)...")
    for i, row in smp_df.iterrows():
        url = row["page_url"]
        if url in done_urls:
            continue
        try:
            rows = process_document("smp", url, row["year"], row["month"])
            all_rows.extend(rows)
            processed_count += 1
            print(f"  [{processed_count}/{total}] OK: {url}")
        except Exception as e:
            print(f"  [{processed_count}/{total}] FAILED: {url} — {e}")
        if processed_count % 10 == 0:
            pd.DataFrame(all_rows).to_csv(OUTPUT_PATH, index=False)
        time.sleep(0.5)

    final_df = pd.DataFrame(all_rows)
    final_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nDone. Saved {len(final_df)} rows (sections) to {OUTPUT_PATH}")
    print(f"Unique documents processed: {final_df['source_url'].nunique()}")


if __name__ == "__main__":
    build_corpus()