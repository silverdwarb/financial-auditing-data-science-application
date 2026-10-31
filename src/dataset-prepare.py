
from pathlib import Path
import pandas as pd


# --------------------------------------------------
# 1. File locations
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SOURCE_FILE = (
    PROJECT_ROOT / "datasets" / "SOT"
    / "journal_entries_4203.csv"
)

RAW_OUTPUT = (
    PROJECT_ROOT / "datasets" / "raw"
    / "journal_entries_raw.csv"
)

ANSWER_KEY_OUTPUT = (
    PROJECT_ROOT / "datasets" / "answers"
    / "journal_entries_answer_key.csv"
)


# --------------------------------------------------
# 2. Define the dataset structure
# --------------------------------------------------

# Original accounting information.
# These columns will be available to the analytics engine.

RAW_COLUMNS = [
    "posting_id",
    "document_date",
    "posting_date",
    "entered_date",
    "gl_account",
    "gl_account_name",
    "amount",
    "cd_flag",
    "user",
    "tax_rate",
    "ref_id",
    "text",
]

# Information excluded from the raw dataset.
# These columns are reserved for reference and evaluation.

ANSWER_KEY_COLUMNS = [
    "promptly",
    "weekend",
    "nwh",
    "label",
    "marking",
    "top_n",
    "high_cash",
]


# --------------------------------------------------
# 3. Load and validate the source dataset
# --------------------------------------------------

def main():

    if not SOURCE_FILE.exists():
        raise FileNotFoundError(
            f"Source dataset not found: {SOURCE_FILE}"
        )

    print("Loading source dataset...")

    df = pd.read_csv(SOURCE_FILE, sep=";")

    print(f"Loaded {len(df):,} posting lines.")
    print(f"Found {len(df.columns)} original columns.")

    required_columns = RAW_COLUMNS + ANSWER_KEY_COLUMNS

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing expected columns: {missing_columns}"
        )

    if "source_row_id" in df.columns:
        raise ValueError(
            "source_row_id already exists in the source dataset."
        )


    # --------------------------------------------------
    # 4. Assign a stable source row identifier
    # --------------------------------------------------

    # Each original posting line receives a unique ID.
    # This is NOT the journal entry identifier.

    df.insert(
        0,
        "source_row_id",
        range(len(df))
    )


    # --------------------------------------------------
    # 5. Create the completely raw dataset
    # --------------------------------------------------

    raw_df = df[
        ["source_row_id"] + RAW_COLUMNS
    ].copy()


    # --------------------------------------------------
    # 6. Create the answer key
    # --------------------------------------------------

    answer_key_df = df[
        ["source_row_id", "posting_id"] + ANSWER_KEY_COLUMNS
    ].copy()


    # --------------------------------------------------
    # 7. Validate the generated datasets
    # --------------------------------------------------

    assert len(raw_df) == len(df)
    assert len(answer_key_df) == len(df)

    assert raw_df["source_row_id"].is_unique
    assert answer_key_df["source_row_id"].is_unique

    assert (
        raw_df["source_row_id"].equals(
            answer_key_df["source_row_id"]
        )
    )

    assert (
        raw_df["posting_id"].equals(
            answer_key_df["posting_id"]
        )
    )

    assert not set(ANSWER_KEY_COLUMNS).intersection(
        raw_df.columns
    )

    assert not set(RAW_COLUMNS).difference(
        raw_df.columns
    )


    # --------------------------------------------------
    # 8. Export both datasets
    # --------------------------------------------------

    RAW_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    ANSWER_KEY_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    raw_df.to_csv(
        RAW_OUTPUT,
        sep=";",
        index=False
    )

    answer_key_df.to_csv(
        ANSWER_KEY_OUTPUT,
        sep=";",
        index=False
    )


    # --------------------------------------------------
    # 9. Print preparation summary
    # --------------------------------------------------

    print("\nDataset preparation successful.")

    print("\nRaw dataset:")
    print(f"  Rows: {len(raw_df):,}")
    print(f"  Columns: {len(raw_df.columns)}")
    print(f"  File: {RAW_OUTPUT}")

    print("\nAnswer key:")
    print(f"  Rows: {len(answer_key_df):,}")
    print(f"  Columns: {len(answer_key_df.columns)}")
    print(f"  File: {ANSWER_KEY_OUTPUT}")

    print("\nSource of Truth was not modified.")


if __name__ == "__main__":
    main()
