"""
CFPB Balanced Dataset Builder v2 — with Label Normalization & Short Labels
===========================================================================

Builds a balanced multi-class dataset from the CFPB Consumer Complaint Database
with two important upgrades over v1:

1. LABEL NORMALIZATION
   The CFPB renamed "Credit reporting or other personal consumer reports" to
   "Credit reporting, credit repair services, or other personal consumer
   reports" in 2022, but BOTH names appear in the database. v1 treated these
   as separate classes. v2 merges them into one canonical category BEFORE
   counting, so they compete fairly with other categories for top-5 selection.

2. SHORT, MEANINGFUL CLASS LABELS
   The raw CFPB labels are unwieldy (e.g., "Checking or savings account").
   v2 maps them to clean short labels for confusion matrices and reports:
       Checking or savings account                                -> Bank Account
       Credit reporting [+ variants]                              -> Credit Reporting
       Debt collection                                            -> Debt Collection
       Mortgage                                                   -> Mortgage
       Credit card or prepaid card                                -> Credit Card
       (and similar mappings for any other top-5 categories)

The output CSV's 'Product' column contains ONLY the short labels, so all
downstream code (the v5 pipeline) just works without changes to label handling.

Output
------
- cfpb_balanced_25k.csv   (5,000 rows per class x 5 classes = 25,000 total)
- cfpb_class_summary.txt  (build report including raw->short label mapping)

Run
---
pip install pandas requests tqdm
python build_cfpb_balanced_dataset_v2.py
"""

import sys
import zipfile
import random
from collections import Counter
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

# ---------- Configuration ----------
CFPB_CSV_ZIP_URL = "https://files.consumerfinance.gov/ccdb/complaints.csv.zip"
ZIP_PATH = Path("complaints.csv.zip")
CSV_PATH = Path("complaints.csv")
OUTPUT_CSV = Path("cfpb_balanced_25k.csv")
SUMMARY_TXT = Path("cfpb_class_summary.txt")

SAMPLES_PER_CLASS = 5000      # 5k per class
NUM_CLASSES = 5               # top 5 distinct categories (post-normalization)
RANDOM_SEED = 42
CHUNK_SIZE = 100_000

USED_COLS = [
    "Date received", "Product", "Sub-product", "Issue",
    "Consumer complaint narrative", "Company", "State",
    "Submitted via", "Company response to consumer", "Timely response?",
]

TEXT_COL = "Consumer complaint narrative"
LABEL_COL = "Product"

# ----- Label normalization map (raw CFPB label -> clean short label) -----
# Keys are matched EXACTLY against the raw 'Product' column. Any raw label
# not in this map is passed through to its own short form via _shorten().
NORMALIZE_MAP = {
    # Both Credit Reporting variants merge into one canonical short label
    "Credit reporting or other personal consumer reports": "Credit Reporting",
    "Credit reporting, credit repair services, or other personal consumer reports": "Credit Reporting",
    "Credit reporting": "Credit Reporting",
    # Bank account
    "Checking or savings account": "Bank Account",
    "Bank account or service": "Bank Account",
    # Credit card
    "Credit card or prepaid card": "Credit Card",
    "Credit card": "Credit Card",
    "Prepaid card": "Credit Card",
    # Loans
    "Debt collection": "Debt Collection",
    "Mortgage": "Mortgage",
    "Student loan": "Student Loan",
    "Vehicle loan or lease": "Vehicle Loan",
    "Consumer Loan": "Consumer Loan",
    "Payday loan, title loan, or personal loan": "Personal Loan",
    "Payday loan, title loan, personal loan, or advance loan": "Personal Loan",
    "Payday loan": "Personal Loan",
    # Money services
    "Money transfer, virtual currency, or money service": "Money Transfer",
    "Money transfers": "Money Transfer",
    "Virtual currency": "Money Transfer",
    "Other financial service": "Other Financial",
}


def normalize_label(raw: str) -> str:
    """
    Map a raw CFPB Product label to its clean short form.
    Falls back to the raw string if not in the map (preserves any new
    categories CFPB might add later).
    """
    if raw is None:
        return None
    raw = raw.strip()
    return NORMALIZE_MAP.get(raw, raw)


# ---------- Step 1: Download ----------
def download_if_needed():
    if CSV_PATH.exists():
        print(f"[OK] Found existing CSV at {CSV_PATH}, skipping download.")
        return
    if ZIP_PATH.exists():
        print(f"[OK] Found existing ZIP at {ZIP_PATH}, skipping download.")
        return

    print(f"[...] Downloading CFPB complaints ZIP from:\n      {CFPB_CSV_ZIP_URL}")
    print("      ~400-500 MB compressed.")
    with requests.get(CFPB_CSV_ZIP_URL, stream=True, timeout=60) as r:
        r.raise_for_status()
        total_size = int(r.headers.get("content-length", 0))
        with open(ZIP_PATH, "wb") as f, tqdm(
            total=total_size, unit="B", unit_scale=True, desc="Downloading"
        ) as pbar:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    print(f"[OK] Downloaded to {ZIP_PATH}")


# ---------- Step 2: Unzip ----------
def unzip_if_needed():
    if CSV_PATH.exists():
        return
    print(f"[...] Extracting {ZIP_PATH} ...")
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        names = z.namelist()
        csv_name = [n for n in names if n.endswith(".csv")][0]
        z.extract(csv_name)
        if csv_name != str(CSV_PATH):
            Path(csv_name).rename(CSV_PATH)
    print(f"[OK] Extracted to {CSV_PATH}")


# ---------- Step 3: Count using normalized labels ----------
def count_normalized_classes():
    """
    Stream the CSV, count rows per NORMALIZED class label (after applying
    the merge map), keeping only rows with a non-empty narrative.
    """
    print(f"[...] First pass: counting NORMALIZED classes with narratives "
          f"(chunk size {CHUNK_SIZE:,})")
    counter = Counter()
    raw_to_normalized_seen = {}  # for the build report
    total_rows = 0
    total_with_text = 0

    reader = pd.read_csv(
        CSV_PATH, usecols=[LABEL_COL, TEXT_COL],
        chunksize=CHUNK_SIZE, dtype=str, low_memory=False,
    )
    for chunk in tqdm(reader, desc="Counting", unit="chunk"):
        total_rows += len(chunk)
        mask = chunk[TEXT_COL].notna() & (chunk[TEXT_COL].str.strip() != "")
        chunk = chunk[mask]
        total_with_text += len(chunk)

        # Normalize labels and count
        labels = chunk[LABEL_COL].dropna().map(normalize_label)
        counter.update(labels.tolist())

        # Track which raw labels mapped to which normalized label
        # (purely for the build report)
        for raw, norm in zip(chunk[LABEL_COL].dropna(),
                             chunk[LABEL_COL].dropna().map(normalize_label)):
            raw_to_normalized_seen.setdefault(norm, set()).add(raw)

    print(f"[OK] Scanned {total_rows:,} rows; "
          f"{total_with_text:,} had narratives.")
    return counter, raw_to_normalized_seen, total_rows, total_with_text


# ---------- Step 4: Collect balanced samples (using normalized labels) ----------
def collect_balanced_sample(top_normalized_classes):
    """
    Stream the CSV again. For each row whose NORMALIZED label is in our top
    set, bucket it under that normalized label. Then sample SAMPLES_PER_CLASS
    from each bucket.
    """
    print(f"[...] Second pass: collecting rows for the top "
          f"{NUM_CLASSES} normalized classes")
    reader = pd.read_csv(
        CSV_PATH, usecols=USED_COLS,
        chunksize=CHUNK_SIZE, dtype=str, low_memory=False,
    )

    buckets = {cls: [] for cls in top_normalized_classes}
    top_set = set(top_normalized_classes)

    for chunk in tqdm(reader, desc="Collecting", unit="chunk"):
        chunk = chunk[
            chunk[TEXT_COL].notna() & (chunk[TEXT_COL].str.strip() != "")
        ]
        if chunk.empty:
            continue

        # Apply normalization. This is the KEY change vs v1.
        chunk = chunk.copy()
        chunk[LABEL_COL] = chunk[LABEL_COL].map(normalize_label)

        # Keep only rows in our chosen top set
        chunk = chunk[chunk[LABEL_COL].isin(top_set)]
        if chunk.empty:
            continue

        for cls, sub in chunk.groupby(LABEL_COL):
            buckets[cls].append(sub)

    print(f"[...] Downsampling each class to exactly "
          f"{SAMPLES_PER_CLASS:,} rows")
    sampled_frames = []
    for cls in top_normalized_classes:
        if not buckets[cls]:
            raise RuntimeError(f"No rows found for normalized class: {cls}")
        full = pd.concat(buckets[cls], ignore_index=True)
        if len(full) < SAMPLES_PER_CLASS:
            raise RuntimeError(
                f"Class '{cls}' only has {len(full):,} rows "
                f"(need {SAMPLES_PER_CLASS:,})."
            )
        sampled = full.sample(
            n=SAMPLES_PER_CLASS, random_state=RANDOM_SEED
        ).reset_index(drop=True)
        sampled_frames.append(sampled)
        print(f"    [OK] {cls:<25s}  "
              f"{len(full):>8,} available -> sampled {len(sampled):,}")

    final_df = pd.concat(sampled_frames, ignore_index=True)
    final_df = final_df.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
    return final_df


# ---------- Step 5: Save outputs ----------
def save_outputs(final_df, class_counter, top_classes, raw_map,
                 total_rows, total_with_text):
    final_df.to_csv(OUTPUT_CSV, index=False)
    print(f"[OK] Wrote balanced dataset -> {OUTPUT_CSV}  "
          f"({len(final_df):,} rows, {len(final_df.columns)} cols)")

    with open(SUMMARY_TXT, "w", encoding="utf-8") as f:
        f.write("CFPB Balanced Dataset Build Summary (v2 with normalization)\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Source: {CFPB_CSV_ZIP_URL}\n")
        f.write(f"Total rows in source CSV:       {total_rows:,}\n")
        f.write(f"Rows with non-empty narrative:  {total_with_text:,}\n\n")

        f.write("NORMALIZED class frequencies (rows with narrative present):\n")
        f.write("-" * 60 + "\n")
        for cls, n in class_counter.most_common():
            marker = "  <-- SELECTED" if cls in top_classes else ""
            f.write(f"  {cls:<35s}  {n:>9,}{marker}\n")

        f.write("\nRaw -> normalized label mapping (only for selected classes):\n")
        f.write("-" * 60 + "\n")
        for cls in top_classes:
            f.write(f"\n  [{cls}]\n")
            for raw in sorted(raw_map.get(cls, [])):
                f.write(f"      <- {raw}\n")

        f.write(f"\nFinal dataset: {OUTPUT_CSV}\n")
        f.write(f"  Classes:       {NUM_CLASSES}\n")
        f.write(f"  Rows/class:    {SAMPLES_PER_CLASS:,}\n")
        f.write(f"  Total rows:    {len(final_df):,}\n")
        f.write(f"  Columns kept:  {', '.join(final_df.columns)}\n")

    print(f"[OK] Wrote summary -> {SUMMARY_TXT}")


# ---------- Main ----------
def main():
    random.seed(RANDOM_SEED)
    download_if_needed()
    unzip_if_needed()

    counter, raw_map, total_rows, total_with_text = count_normalized_classes()
    if not counter:
        print("[ERR] No classes with narratives found.", file=sys.stderr)
        sys.exit(1)

    top_classes = [cls for cls, _ in counter.most_common(NUM_CLASSES)]
    print("\nTop classes (normalized, by # of complaints with narrative):")
    for i, cls in enumerate(top_classes, 1):
        print(f"  {i}. {cls:<25s}  ({counter[cls]:,} rows available)")
    print()

    final_df = collect_balanced_sample(top_classes)
    save_outputs(final_df, counter, top_classes, raw_map,
                 total_rows, total_with_text)

    print("\n[DONE] Balanced dataset ready:")
    print(f"       {OUTPUT_CSV.resolve()}")
    print(f"\nClass labels in the output:")
    for cls in top_classes:
        print(f"       - {cls}")


if __name__ == "__main__":
    main()