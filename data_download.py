"""
CFPB Balanced Dataset Builder v3 — Label Normalization + "Other" Class for OOD
===============================================================================

Builds a balanced multi-class dataset from the CFPB Consumer Complaint Database
with three upgrades:

1. LABEL NORMALIZATION
   The CFPB renamed "Credit reporting or other personal consumer reports" to
   "Credit reporting, credit repair services, or other personal consumer
   reports" in 2022, but BOTH names appear in the database. v2+ merges them
   into one canonical category BEFORE counting so they compete fairly with
   other categories for top-N selection.

2. SHORT, MEANINGFUL CLASS LABELS
   Raw CFPB labels are unwieldy (e.g., "Checking or savings account"). They
   are mapped to clean short labels for confusion matrices and reports.

3. "OTHER" CLASS for OUT-OF-SCOPE DETECTION
   When INCLUDE_OTHER_CLASS=True, a 6th "Other" class is built by stratified
   sampling across all DISCARDED normalized classes (Money Transfer, Student
   Loan, Vehicle Loan, Personal Loan, Consumer Loan, Debt mgmt, Other
   Financial). This lets a downstream classifier explicitly recognize "this
   is a CFPB-style complaint but not one of my 5 in-scope topics" instead of
   being forced to pick the closest top-5 class. Stratified sampling prevents
   any single discarded class (e.g., Money Transfer with 116k rows) from
   dominating the Other bucket — each contributes its fair share up to
   availability, with deficits redistributed to larger pools.

Output
------
- cfpb_with_other_30k.csv   (5k/class x 5 + 5k Other = 30,000 total)  [default]
  or cfpb_balanced_25k.csv  (5k/class x 5 = 25,000 total)             [legacy]
- cfpb_class_summary.txt    (build report incl. raw->short label mapping)

Run
---
pip install pandas requests tqdm
python data_download.py
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
SUMMARY_TXT = Path("cfpb_class_summary.txt")

SAMPLES_PER_CLASS = 5000      # 5k per class
NUM_CLASSES = 5               # top 5 distinct categories (post-normalization)
RANDOM_SEED = 42
CHUNK_SIZE = 100_000

# ---- Out-of-Scope (Other) class config ----
# When True, build a 6th "Other" class via stratified sampling across all
# DISCARDED normalized classes (Money Transfer, Student Loan, Vehicle Loan,
# Personal Loan, Consumer Loan, Debt mgmt, Other Financial). This gives the
# downstream classifier an explicit way to recognize complaints that aren't
# in the top-5 in-scope set, instead of being forced to pick the closest
# in-scope class.
INCLUDE_OTHER_CLASS = True
OTHER_CLASS_NAME = "Other"
SAMPLES_OTHER = SAMPLES_PER_CLASS  # match per-class size for balanced training

OUTPUT_CSV = (
    Path("cfpb_with_other_30k.csv") if INCLUDE_OTHER_CLASS
    else Path("cfpb_balanced_25k.csv")
)

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
def collect_balanced_sample(top_normalized_classes, include_other=False):
    """
    Stream the CSV again. For each row whose NORMALIZED label is in our top
    set, bucket it under that normalized label. Then sample SAMPLES_PER_CLASS
    from each bucket.

    If include_other=True, also bucket rows from non-top normalized classes
    so we can later build a stratified "Other" class for OOD detection.
    """
    msg = f"[...] Second pass: collecting rows for the top {NUM_CLASSES} normalized classes"
    if include_other:
        msg += f" + '{OTHER_CLASS_NAME}' class ({SAMPLES_OTHER:,} rows from discarded classes)"
    print(msg)

    reader = pd.read_csv(
        CSV_PATH, usecols=USED_COLS,
        chunksize=CHUNK_SIZE, dtype=str, low_memory=False,
    )

    buckets = {cls: [] for cls in top_normalized_classes}
    other_buckets = {}  # normalized_label -> list of subframes (only if include_other)
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

        # Top classes
        in_top = chunk[chunk[LABEL_COL].isin(top_set)]
        for cls, sub in in_top.groupby(LABEL_COL):
            buckets[cls].append(sub)

        # Discarded classes (for the Other bucket)
        if include_other:
            in_other = chunk[~chunk[LABEL_COL].isin(top_set) & chunk[LABEL_COL].notna()]
            for cls, sub in in_other.groupby(LABEL_COL):
                other_buckets.setdefault(cls, []).append(sub)

    print(f"[...] Downsampling each top class to exactly "
          f"{SAMPLES_PER_CLASS:,} rows")
    sampled_frames = []
    other_composition = {}  # for the build report
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

    if include_other and other_buckets:
        other_df, other_composition = build_other_class(other_buckets, SAMPLES_OTHER)
        sampled_frames.append(other_df)
        print(f"    [OK] {OTHER_CLASS_NAME:<25s}  -> "
              f"sampled {len(other_df):,} from {len(other_buckets)} discarded classes")

    final_df = pd.concat(sampled_frames, ignore_index=True)
    final_df = final_df.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
    return final_df, other_composition


def build_other_class(other_buckets, total_samples):
    """
    Build the 'Other' class via stratified sampling across discarded classes.

    Each discarded class contributes equally up to its availability (round 1).
    Any deficit (from classes with too few rows) plus the remainder from
    integer division is filled in round 2 by uniformly sampling the union of
    leftover rows across all classes that had surplus. This prevents one big
    class (e.g. Money Transfer with 116k rows) from dominating the Other
    bucket while still using all available data.

    Returns
    -------
    other_df : pd.DataFrame
        DataFrame with LABEL_COL set to OTHER_CLASS_NAME for every row.
    composition : dict[str, int]
        Final per-source-class counts, for the build report.
    """
    concatenated = {
        cls: pd.concat(frames, ignore_index=True)
        for cls, frames in other_buckets.items()
    }

    n_classes = len(concatenated)
    base_quota = total_samples // n_classes
    remainder = total_samples - base_quota * n_classes

    sampled_frames = []
    composition = {}
    surplus_pools = {}
    deficit = remainder

    # Round 1: equal-quota draw from each discarded class
    for cls, df in concatenated.items():
        n_avail = len(df)
        take = min(base_quota, n_avail)
        if take > 0:
            sampled = df.sample(n=take, random_state=RANDOM_SEED)
            sampled_frames.append(sampled)
            composition[cls] = take
            leftover = df.drop(sampled.index)
            if len(leftover) > 0:
                surplus_pools[cls] = leftover
        else:
            composition[cls] = 0
        if n_avail < base_quota:
            deficit += (base_quota - n_avail)

    # Round 2: redistribute deficit + remainder across surplus pools (uniform)
    if deficit > 0 and surplus_pools:
        surplus_pool = pd.concat(surplus_pools.values(), ignore_index=True)
        n_take = min(deficit, len(surplus_pool))
        extra = surplus_pool.sample(n=n_take, random_state=RANDOM_SEED)
        sampled_frames.append(extra)
        # Update composition with round-2 contributions
        for cls, count in extra[LABEL_COL].value_counts().items():
            composition[cls] = composition.get(cls, 0) + int(count)

    other_df = pd.concat(sampled_frames, ignore_index=True)
    # Defensive truncate (shouldn't trigger, but harmless)
    if len(other_df) > total_samples:
        other_df = other_df.sample(n=total_samples, random_state=RANDOM_SEED)
    other_df = other_df.reset_index(drop=True)
    other_df[LABEL_COL] = OTHER_CLASS_NAME
    return other_df, composition


# ---------- Step 5: Save outputs ----------
def save_outputs(final_df, class_counter, top_classes, raw_map,
                 total_rows, total_with_text, other_composition=None):
    final_df.to_csv(OUTPUT_CSV, index=False)
    print(f"[OK] Wrote balanced dataset -> {OUTPUT_CSV}  "
          f"({len(final_df):,} rows, {len(final_df.columns)} cols)")

    with open(SUMMARY_TXT, "w", encoding="utf-8") as f:
        f.write("CFPB Balanced Dataset Build Summary (v3 with Other-class for OOD)\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Source: {CFPB_CSV_ZIP_URL}\n")
        f.write(f"Total rows in source CSV:       {total_rows:,}\n")
        f.write(f"Rows with non-empty narrative:  {total_with_text:,}\n\n")

        f.write("NORMALIZED class frequencies (rows with narrative present):\n")
        f.write("-" * 60 + "\n")
        for cls, n in class_counter.most_common():
            if cls in top_classes:
                marker = "  <-- SELECTED (in-scope)"
            elif INCLUDE_OTHER_CLASS:
                marker = "  <-- folded into 'Other'"
            else:
                marker = ""
            f.write(f"  {cls:<35s}  {n:>9,}{marker}\n")

        f.write("\nRaw -> normalized label mapping (only for selected classes):\n")
        f.write("-" * 60 + "\n")
        for cls in top_classes:
            f.write(f"\n  [{cls}]\n")
            for raw in sorted(raw_map.get(cls, [])):
                f.write(f"      <- {raw}\n")

        if INCLUDE_OTHER_CLASS and other_composition:
            f.write("\n'Other' class composition (stratified across discards):\n")
            f.write("-" * 60 + "\n")
            for cls, n in sorted(other_composition.items(), key=lambda x: -x[1]):
                f.write(f"  {cls:<35s}  {n:>5,} rows\n")

        f.write(f"\nFinal dataset: {OUTPUT_CSV}\n")
        n_classes_out = NUM_CLASSES + (1 if INCLUDE_OTHER_CLASS else 0)
        f.write(f"  Classes:       {n_classes_out}"
                f"{' (5 in-scope + Other)' if INCLUDE_OTHER_CLASS else ''}\n")
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

    final_df, other_composition = collect_balanced_sample(
        top_classes, include_other=INCLUDE_OTHER_CLASS
    )
    save_outputs(final_df, counter, top_classes, raw_map,
                 total_rows, total_with_text, other_composition)

    print("\n[DONE] Balanced dataset ready:")
    print(f"       {OUTPUT_CSV.resolve()}")
    print(f"\nClass labels in the output:")
    for cls in top_classes:
        print(f"       - {cls}")
    if INCLUDE_OTHER_CLASS:
        print(f"       - {OTHER_CLASS_NAME}  (out-of-scope sentinel)")


if __name__ == "__main__":
    main()