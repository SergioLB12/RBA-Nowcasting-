"""
NLP feature pipeline: takes data/processed/text_corpus.csv (built by
build_corpus.py) and produces one row per quarter with:
  - a sentence embedding (averaged across all text for that quarter)
  - a Loughran-McDonald sentiment score (averaged across all text for that quarter)

Run this locally (needs internet access the first time, to download the
sentence-transformers model — after that it runs offline).

Requirements:
    pip install sentence-transformers pandas numpy

You also need the Loughran-McDonald Master Dictionary CSV. Download it from:
    https://sraf.nd.edu/loughranmcdonald-master-dictionary/
(free, no registration needed for the CSV itself — look for
"Loughran-McDonald Master Dictionary w/ Sentiment Word Lists").
Save it as: data/external/loughran_mcdonald_dict.csv
"""

import re
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

TEXT_CORPUS_PATH = "../data/processed/text_corpus.csv"
LM_DICT_PATH = "../data/external/loughran_mcdonald_dict.csv"
MACRO_PATH = "../data/processed/macro_clean.xlsx"
OUTPUT_PATH = "../data/processed/quarterly_features.csv"

# SMP publication month -> the quarter it actually analyses (published ~1-2
# months after the quarter it reports on; see project notes).
SMP_MONTH_TO_QUARTER_OFFSET = {
    "feb": ("Q4", -1),   # Feb SMP analyses Q4 of the PREVIOUS year
    "may": ("Q1", 0),
    "aug": ("Q2", 0),
    "nov": ("Q3", 0),
}


def extract_month_from_minutes_url(url: str) -> int | None:
    """Extract the meeting month from a minutes URL, handling both the old
    (DDMMYYYY.html) and new (YYYY-MM-DD.html) formats."""
    m = re.search(r"/(\d{4})-(\d{2})-(\d{2})\.html", url)
    if m:
        return int(m.group(2))
    m = re.search(r"/(\d{2})(\d{2})(\d{4})\.html", url)
    if m:
        return int(m.group(2))
    return None


def month_to_calendar_quarter(month: int) -> str:
    return f"Q{(month - 1) // 3 + 1}"


def assign_quarter_label(row) -> str | None:
    """Return a 'YYYY-Qn' label for a document row."""
    if row["doc_type"] == "minutes":
        month = extract_month_from_minutes_url(row["source_url"])
        if month is None:
            return None
        return f"{row['year']}-{month_to_calendar_quarter(month)}"
    else:  # smp
        quarter, year_offset = SMP_MONTH_TO_QUARTER_OFFSET[row["month"]]
        return f"{row['year'] + year_offset}-{quarter}"


def load_lm_dictionary(path: str) -> tuple[set, set]:
    """Load the Loughran-McDonald dictionary and return (positive_words, negative_words)."""
    lm = pd.read_csv(path)
    lm.columns = [c.strip() for c in lm.columns]
    positive = set(lm.loc[lm["Positive"] > 0, "Word"].str.lower())
    negative = set(lm.loc[lm["Negative"] > 0, "Word"].str.lower())
    return positive, negative


def lm_sentiment_score(text: str, positive: set, negative: set) -> float:
    """(#positive words - #negative words) / total words."""
    words = re.findall(r"[a-zA-Z']+", text.lower())
    if not words:
        return 0.0
    pos_count = sum(1 for w in words if w in positive)
    neg_count = sum(1 for w in words if w in negative)
    return (pos_count - neg_count) / len(words)


def main():
    print("Loading text corpus...")
    corpus = pd.read_csv(TEXT_CORPUS_PATH)
    corpus["quarter"] = corpus.apply(assign_quarter_label, axis=1)
    corpus = corpus.dropna(subset=["quarter", "text"])
    print(f"  {len(corpus)} rows mapped to quarters ({corpus['quarter'].nunique()} unique quarters)")

    print("Loading sentence-transformers model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Computing embeddings (this may take a few minutes)...")
    corpus["text"] = corpus["text"].astype(str)
    embeddings = model.encode(corpus["text"].tolist(), show_progress_bar=True)
    embedding_cols = [f"emb_{i}" for i in range(embeddings.shape[1])]
    emb_df = pd.DataFrame(embeddings, columns=embedding_cols)
    corpus = pd.concat([corpus.reset_index(drop=True), emb_df], axis=1)

    print("Loading Loughran-McDonald dictionary and computing sentiment...")
    positive, negative = load_lm_dictionary(LM_DICT_PATH)
    corpus["sentiment"] = corpus["text"].apply(lambda t: lm_sentiment_score(t, positive, negative))

    print("Aggregating to quarterly level (mean across all chunks/sections)...")
    agg_dict = {c: "mean" for c in embedding_cols}
    agg_dict["sentiment"] = "mean"
    quarterly = corpus.groupby("quarter").agg(agg_dict).reset_index()
    quarterly["n_documents"] = corpus.groupby("quarter")["source_url"].nunique().values

    print(f"  {len(quarterly)} quarters with text features")

    print("Merging with macro dataset...")
    macro = pd.read_excel(MACRO_PATH)
    macro_col = "quarter" if "quarter" in macro.columns else macro.columns[0]
    final = macro.merge(quarterly, left_on=macro_col, right_on="quarter", how="left")

    final.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved {OUTPUT_PATH}")
    print(f"Quarters with macro data: {len(macro)}")
    print(f"Quarters with text features matched: {final['n_documents'].notna().sum()}")


if __name__ == "__main__":
    main()