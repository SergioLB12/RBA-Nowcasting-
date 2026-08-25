# Text-Augmented Nowcasting of Inflation in Australia

Data Science Capstone — Master of Data Science, Sydney Polytechnic Institute

## Research Question

Does incorporating textual features extracted from RBA communications (Statements on
Monetary Policy and Board minutes) improve the accuracy of quarterly inflation
nowcasting in Australia, compared to classical econometric models and machine
learning models based solely on hard macroeconomic variables?

## Specifications compared

1. **Bridge equation** — classical econometric baseline
2. **Random Forest** — macroeconomic variables only
3. **Random Forest + Text** — same model, augmented with sentence embeddings
   (`all-MiniLM-L6-v2`) and Loughran-McDonald sentiment scores extracted from
   RBA Statements on Monetary Policy and Board minutes

## Data

- **Target**: quarterly CPI (ABS, re-referenced series), 2007–2026 (~78 observations)
- **Macro predictors**: cash rate (RBA Table F1), labour market (ABS)
- **Text corpus**: ~78 Statements on Monetary Policy + ~213 Board minutes (~291 documents)

## Repository structure

```
src/
  discover_documents.py   # crawls RBA site, builds document URL index
  download_dataset.py     # downloads macro data (CPI, cash rate)
  text_preprocessing.py   # cleans + chunks SMP/minutes text
  nlp_features.py         # (WIP) embeddings + sentiment feature generation
  models/                 # (WIP) Bridge equation, Random Forest, Random Forest+Text
notebooks/
  01_eda.ipynb
  02_model_comparison.ipynb
  03_results.ipynb
data/
  external/                # small index CSVs (committed)
  raw/                      # scraped text, downloaded macro files (gitignored)
  processed/                 # cleaned datasets (gitignored)
docs/
  proposal.docx
  literature_matrix.xlsx
```

## Setup

```bash
conda create -n rba-nowcasting python=3.11
conda activate rba-nowcasting
pip install -r requirements.txt
```

## Pipeline order

1. `python src/discover_documents.py` — builds `minutes_index.csv` / `smp_index.csv`
2. `python src/download_dataset.py` — downloads macro data
3. Run `text_preprocessing.py` functions over the discovered URLs to build the
   cleaned text corpus
4. `notebooks/01_eda.ipynb` — explore the collected data
5. Model development (Weeks 5–7 of project timeline)

## Notes

- RBA/ABS data and text are used under the RBA's public/personal-use copyright
  terms — raw and processed text are not committed to this repository; only the
  scripts to regenerate them are.
