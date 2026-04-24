"""
Build a balanced 5-class subset of the CFPB Consumer Complaints Database
=========================================================================

What this script does
---------------------
1. Downloads the full CFPB Consumer Complaint Database (CSV, ~1 GB uncompressed).
2. Streams through the file in chunks (so you don't need 8 GB of RAM).
3. Keeps only rows that have a non-empty narrative (text).
4. Counts Product categories and identifies the TOP 5 most common classes.
5. Samples exactly 1,000 records from each of those 5 classes (5,000 total).
6. Saves a clean, balanced CSV ready for TF-IDF / multi-class classification.

Output
------
- cfpb_balanced_5k.csv  (the final dataset your project will use)
- cfpb_class_summary.txt (a small report on the classes chosen and counts)

Requirements
------------
pip install pandas requests tqdm

Run
---
python build_cfpb_balanced_dataset.py

Runtime: ~3-8 minutes depending on your internet and disk speed.
Disk usage during run: ~1.2 GB temporarily, ~5 MB final output.
"""

import os
import sys
import zipfile
import io
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
OUTPUT_CSV = Path("cfpb_balanced_5k.csv")
SUMMARY_TXT = Path("cfpb_class_summary.txt")

SAMPLES_PER_CLASS = 1000      # 1k per class
NUM_CLASSES = 5               # top 5 classes
RANDOM_SEED = 42              # reproducibility
CHUNK_SIZE = 100_000          # rows to process at a time (memory friendly)

# Columns we actually need. Everything else is ignored.
# (The full CSV has ~18 columns; we only keep these.)
USED_COLS = [
    "Date received",
    "Product",
    "Sub-product",
    "Issue",
    "Consumer complaint narrative",
    "Company",
    "State",
    "Submitted via",
    "Company response to consumer",
    "Timely response?",
]

# Narrative field — this is what we'll classify
TEXT_COL = "Consumer complaint narrative"
LABEL_COL = "Product"


# ---------- Step 1: Download ----------
def download_if_needed():
    """Download the CFPB ZIP only if we don't already have it or the CSV."""
    if CSV_PATH.exists():
        print(f"[OK] Found existing CSV at {CSV_PATH}, skipping download.")
        return
    if ZIP_PATH.exists():
        print(f"[OK] Found existing ZIP at {ZIP_PATH}, skipping download.")
        return

    print(f"[...] Downloading CFPB complaints ZIP from:\n      {CFPB_CSV_ZIP_URL}")
    print("      This is ~400-500 MB compressed. Grab a coffee.")

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
        # The archive contains a single file, usually 'complaints.csv'
        names = z.namelist()
        csv_name = [n for n in names if n.endswith(".csv")][0]
        z.extract(csv_name)
        # Rename to our standard path if needed
        if csv_name != str(CSV_PATH):
            Path(csv_name).rename(CSV_PATH)
    print(f"[OK] Extracted to {CSV_PATH}")


# ---------- Step 3: First pass — count class sizes (text-only rows) ----------
def count_classes_with_narrative():
    """
    Stream through the CSV and count how many rows each Product has
    WHEN the narrative is non-empty. This avoids keeping classes that
    technically exist but have almost no usable text.
    """
    print(f"[...] First pass: counting classes with non-empty narratives (chunk size {CHUNK_SIZE:,})")
    counter = Counter()
    total_rows_seen = 0
    total_with_text = 0

    # Only load the two columns we need for counting -> much faster
    reader = pd.read_csv(
        CSV_PATH,
        usecols=[LABEL_COL, TEXT_COL],
        chunksize=CHUNK_SIZE,
        dtype=str,               # treat everything as string (safer)
        low_memory=False,
    )

    for chunk in tqdm(reader, desc="Counting", unit="chunk"):
        total_rows_seen += len(chunk)
        # Keep rows that actually have a narrative (non-null, not whitespace)
        mask = chunk[TEXT_COL].notna() & (chunk[TEXT_COL].str.strip() != "")
        chunk = chunk[mask]
        total_with_text += len(chunk)
        counter.update(chunk[LABEL_COL].dropna().tolist())

    print(f"[OK] Scanned {total_rows_seen:,} total rows; {total_with_text:,} had narratives.")
    return counter, total_rows_seen, total_with_text


# ---------- Step 4: Second pass — collect samples ----------
def collect_balanced_sample(top_classes):
    """
    Stream through the file again and collect EVERY matching row for the top
    classes (with narrative present). We then randomly downsample each class
    to SAMPLES_PER_CLASS at the end.

    Memory note: even keeping every matching row, this is fine because the top
    5 classes with narratives are typically under a few hundred thousand rows
    and only a few columns wide.
    """
    print(f"[...] Second pass: collecting all rows from the {NUM_CLASSES} chosen classes")
    # Use only the columns we care about (keeps memory down)
    reader = pd.read_csv(
        CSV_PATH,
        usecols=USED_COLS,
        chunksize=CHUNK_SIZE,
        dtype=str,
        low_memory=False,
    )

    buckets = {cls: [] for cls in top_classes}
    top_set = set(top_classes)

    for chunk in tqdm(reader, desc="Collecting", unit="chunk"):
        # Drop empty narratives
        chunk = chunk[
            chunk[TEXT_COL].notna() & (chunk[TEXT_COL].str.strip() != "")
        ]
        # Keep only rows whose Product is one of our top 5
        chunk = chunk[chunk[LABEL_COL].isin(top_set)]
        if chunk.empty:
            continue
        # Split into per-class buckets
        for cls, sub in chunk.groupby(LABEL_COL):
            buckets[cls].append(sub)

    # Concat per-class frames, then sample 1,000 from each
    print("[...] Downsampling each class to exactly "
          f"{SAMPLES_PER_CLASS:,} rows")
    rng = random.Random(RANDOM_SEED)
    sampled_frames = []
    for cls in top_classes:
        if not buckets[cls]:
            raise RuntimeError(f"No rows found for class: {cls}")
        full = pd.concat(buckets[cls], ignore_index=True)
        if len(full) < SAMPLES_PER_CLASS:
            raise RuntimeError(
                f"Class '{cls}' only has {len(full)} rows (need "
                f"{SAMPLES_PER_CLASS}). Pick a different class or "
                f"lower SAMPLES_PER_CLASS."
            )
        # Deterministic random sample
        sampled = full.sample(
            n=SAMPLES_PER_CLASS, random_state=RANDOM_SEED
        ).reset_index(drop=True)
        sampled_frames.append(sampled)
        print(f"    [OK] {cls}: {len(full):,} available -> sampled {len(sampled):,}")

    final_df = pd.concat(sampled_frames, ignore_index=True)
    # Shuffle rows so classes aren't in blocks
    final_df = final_df.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
    return final_df


# ---------- Step 5: Save outputs ----------
def save_outputs(final_df, class_counter, top_classes, total_rows, total_with_text):
    final_df.to_csv(OUTPUT_CSV, index=False)
    print(f"[OK] Wrote balanced dataset -> {OUTPUT_CSV}  "
          f"({len(final_df):,} rows, {len(final_df.columns)} cols)")

    # Human-readable summary
    with open(SUMMARY_TXT, "w", encoding="utf-8") as f:
        f.write("CFPB Balanced Dataset Build Summary\n")
        f.write("=" * 42 + "\n\n")
        f.write(f"Source: {CFPB_CSV_ZIP_URL}\n")
        f.write(f"Total rows in source CSV:           {total_rows:,}\n")
        f.write(f"Rows with non-empty narrative:      {total_with_text:,}\n\n")

        f.write("Class frequencies (rows with narrative present):\n")
        f.write("-" * 42 + "\n")
        for cls, n in class_counter.most_common():
            marker = "  <-- SELECTED" if cls in top_classes else ""
            f.write(f"  {cls:<55}  {n:>8,}{marker}\n")

        f.write("\n")
        f.write(f"Final dataset: {OUTPUT_CSV}\n")
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

    counter, total_rows, total_with_text = count_classes_with_narrative()

    if not counter:
        print("[ERR] No classes with narratives found. Aborting.", file=sys.stderr)
        sys.exit(1)

    top_classes = [cls for cls, _ in counter.most_common(NUM_CLASSES)]
    print("\nTop classes (by # of complaints with narrative):")
    for i, cls in enumerate(top_classes, 1):
        print(f"  {i}. {cls}  ({counter[cls]:,} rows available)")
    print()

    final_df = collect_balanced_sample(top_classes)
    save_outputs(final_df, counter, top_classes, total_rows, total_with_text)

    print("\n[DONE] You now have a clean balanced dataset:")
    print(f"       {OUTPUT_CSV.resolve()}")
    print("\nNext steps for your project:")
    print("  - Load with: df = pd.read_csv('cfpb_balanced_5k.csv')")
    print("  - Target column:  'Product'")
    print("  - Text column:    'Consumer complaint narrative'")
    print("  - Extra features: 'Sub-product', 'Issue', 'State', "
          "'Submitted via', 'Company response to consumer', 'Timely response?'")


if __name__ == "__main__":
    main()