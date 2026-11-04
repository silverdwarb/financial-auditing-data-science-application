import pandas as pd

from src.data_loader import load_dataset
from src.forensic_tests.add_metadata import add_metadata


def main():
    print("Loading journal entry dataset...")

    df = load_dataset()

    print(f"Original posting lines: {len(df):,}")
    print(f"Original journal entries: {df['posting_id'].nunique():,}")

    print("\nGenerating metadata...")

    result = add_metadata(df)

    print("\nMETADATA GENERATION COMPLETE")

    # --------------------------------------------------
    # 1. Dataset integrity
    # --------------------------------------------------

    print("\n--- DATASET INTEGRITY ---")

    print(f"Original rows: {len(df):,}")
    print(f"Result rows:   {len(result):,}")

    print(
        "Source row IDs preserved:",
        result["source_row_id"].is_unique
        and set(result["source_row_id"]) == set(df["source_row_id"])
    )

    print(
        "Original columns preserved:",
        all(column in result.columns for column in df.columns)
    )

    # --------------------------------------------------
    # 2. Metadata completeness
    # --------------------------------------------------

    metadata_columns = [
        "involves_cash",
        "high_cash",
        "weekend_activity",
        "after_hours",
        "posting_delay_days",
        "round_10",
        "round_100",
        "round_1000",
    ]

    print("\n--- METADATA COMPLETENESS ---")

    for column in metadata_columns:
        missing = result[column].isna().sum()

        print(
            f"{column:<25} "
            f"Missing: {missing:,}"
        )

    # --------------------------------------------------
    # 3. Metadata flag counts
    # --------------------------------------------------

    flag_columns = [
        "involves_cash",
        "high_cash",
        "weekend_activity",
        "after_hours",
        "round_10",
        "round_100",
        "round_1000",
    ]

    print("\n--- METADATA FLAG COUNTS ---")

    for column in flag_columns:
        count = result[column].sum()
        percentage = count / len(result) * 100

        print(
            f"{column:<25} "
            f"{count:>10,} "
            f"({percentage:.2f}%)"
        )

    # --------------------------------------------------
    # 4. Posting delay summary
    # --------------------------------------------------

    print("\n--- POSTING DELAY SUMMARY ---")

    print(
        result["posting_delay_days"].describe().round(2)
    )

    # --------------------------------------------------
    # 5. Journal-entry-level counts
    # --------------------------------------------------

    print("\n--- UNIQUE JOURNAL ENTRY COUNTS ---")

    entry_level_columns = [
        "involves_cash",
        "high_cash",
        "weekend_activity",
        "after_hours",
    ]

    for column in entry_level_columns:
        entry_flags = result.groupby("posting_id")[column].first()

        print(
            f"{column:<25} "
            f"{entry_flags.sum():>10,} "
            f"of {len(entry_flags):,} entries"
        )

    # --------------------------------------------------
    # 6. Export
    # --------------------------------------------------

    output_path = "datasets/raw/journal_entries_with_metadata.csv"

    result.to_csv(
        output_path,
        sep=";",
        index=False
    )

    print(f"\nExported results to: {output_path}")


if __name__ == "__main__":
    main()