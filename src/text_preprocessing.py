"""
Text preprocessing pipeline for RBA Statements on Monetary Policy (SMP) and
Board minutes.

Run this locally (needs internet access) — not inside this chat environment.
    pip install requests beautifulsoup4 pandas

What it does:
1. Fetches a document's HTML page.
2. Strips navigation/menu boilerplate, keeping only the substantive text
   (from the first content heading to the end of the last content section).
3. Cleans whitespace and normalises encoding artefacts (smart quotes, en-dashes).
4. For long documents (SMP), splits into named sections and returns
   chunk-level text so each chunk fits within typical sentence-embedding
   token limits (~256-384 tokens for all-MiniLM-L6-v2).
5. Computes basic length statistics per document (words, characters) —
   useful for the EDA / dataset-description slide.
"""

import re
import requests
from bs4 import BeautifulSoup
import pandas as pd

# Section headers used by both SMP and Minutes documents (adjust as needed
# once you inspect a full year's worth of pages — formatting has changed
# somewhat over 2007-2026).
MINUTES_SECTION_HEADERS = [
    "Financial conditions", "Economic conditions", "Economic outlook",
    "Considerations for monetary policy", "The decision",
]
SMP_SECTION_HEADERS = [
    "Overview", "International economic conditions", "Domestic economic conditions",
    "Domestic financial conditions", "Inflation", "Economic outlook",
]


def fetch_clean_text(url: str) -> str:
    """Download a page and return cleaned body text (boilerplate stripped)."""
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.content, "html.parser")

    # RBA content pages typically wrap the substantive text in <main> or a
    # div with class 'content'. Adjust selector after inspecting real pages.
    main = soup.find("main") or soup.find("div", {"class": "content"}) or soup

    # Drop nav, header, footer, script/style elements.
    for tag in main.find_all(["nav", "header", "footer", "script", "style"]):
        tag.decompose()

    text = main.get_text(separator="\n")
    text = clean_text(text)
    return text


def clean_text(text: str) -> str:
    """Normalise whitespace and typographic characters."""
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = "\n".join(line.strip() for line in text.split("\n") if line.strip())
    return text.strip()


def split_into_sections(text: str, section_headers: list[str]) -> dict[str, str]:
    """Split a cleaned document into named sections for chunk-level embeddings.
    Handles both plain headers on their own line and markdown '## Header' style."""
    pattern = "|".join(re.escape(h) for h in section_headers)
    matches = list(re.finditer(rf"^#{{0,3}}\s*({pattern})\s*$", text, flags=re.MULTILINE))

    if not matches:
        return {"Full document": text}

    sections = {}
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections[m.group(1)] = text[start:end].strip()
    return sections


def document_stats(doc_id: str, text: str) -> dict:
    """Length statistics for the EDA slide."""
    words = len(text.split())
    chars = len(text)
    chars_no_space = len(text.replace(" ", "").replace("\n", ""))
    return {
        "doc_id": doc_id,
        "n_words": words,
        "n_chars": chars,
        "n_chars_no_space": chars_no_space,
    }


if __name__ == "__main__":
    # Example: one minutes URL and one SMP URL (replace with real archive URLs
    # once you have the full list from the RBA site index).
    example_urls = {
        "minutes_2026_05": "https://www.rba.gov.au/monetary-policy/rba-board-minutes/2026/2026-05-05.html",
        # "smp_2026_05": "https://www.rba.gov.au/publications/smp/2026/may/",
    }

    stats_rows = []
    for doc_id, url in example_urls.items():
        text = fetch_clean_text(url)
        stats_rows.append(document_stats(doc_id, text))

        sections = split_into_sections(text, MINUTES_SECTION_HEADERS)
        print(f"\n{doc_id}: {len(sections)} sections found")
        for name, content in sections.items():
            print(f"  - {name}: {len(content.split())} words")

    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv("text_corpus_stats.csv", index=False)
    print("\nSaved text_corpus_stats.csv")
