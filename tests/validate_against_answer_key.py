
from pathlib import Path

import pandas as pd


# --------------------------------------------------
# FILE LOCATIONS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

GENERATED_FILE = (
    PROJECT_ROOT
    / "datasets"
    / "raw"
    / "journal_entries_with_metadata.csv"
)

ANSWER_KEY_FILE = (
    PROJECT_ROOT
    / "datasets"
    / "answers"
    / "journal_entries_answer_key.csv"
)

MISMATCH_OUTPUT_FILE = (
    PROJECT_ROOT
    / "datasets"
    / "answers"
    / "answer_key_mismatches.csv"
)


# --------------------------------------------------
# DATA LOADING
# --------------------------------------------------

def load_comparison_files():
    generated = pd.read_csv(
        GENERATED_FILE,
        sep=";",
        dtype={
            "source_row_id": "string",
            "posting_id": "string",
        },
    )

    answer_key = pd.read_csv(
        ANSWER_KEY_FILE,
        sep=";",
        dtype={
            "source_row_id": "string",
            "posting_id": "string",
        },
    )

    return generated, answer_key


# --------------------------------------------------
# STRUCTURAL VALIDATION
# --------------------------------------------------

def validate_structure(generated, answer_key):
    required_generated = [
        "source_row_id",
        "posting_id",
        "entered_date",
        "after_hours",
        "weekend_activity",
        "promptly",
        "high_cash",
    ]

    required_reference = [
        "source_row_id",
        "posting_id",
        "promptly",
        "weekend",
        "nwh",
        "label",
        "marking",
        "top_n",
        "high_cash",
    ]

    for column in required_generated:
        if column not in generated.columns:
            raise ValueError(
                f"Generated metadata is missing column: {column}"
            )

    for column in required_reference:
        if column not in answer_key.columns:
            raise ValueError(
                f"Answer key is missing column: {column}"
            )

    for name, df in [
        ("Generated metadata", generated),
        ("Answer key", answer_key),
    ]:
        if df["source_row_id"].isna().any():
            raise ValueError(
                f"{name} contains missing source_row_id values."
            )

        if df["source_row_id"].duplicated().any():
            raise ValueError(
                f"{name} contains duplicate source_row_id values."
            )

    generated_ids = set(generated["source_row_id"])
    reference_ids = set(answer_key["source_row_id"])

    generated_only = generated_ids - reference_ids
    reference_only = reference_ids - generated_ids

    print("\n--- ROW ALIGNMENT ---")
    print(f"Generated rows: {len(generated):,}")
    print(f"Answer key rows: {len(answer_key):,}")
    print(f"Rows only in generated file: {len(generated_only):,}")
    print(f"Rows only in answer key: {len(reference_only):,}")

    if generated_only or reference_only:
        print("\nWARNING: Source row IDs do not completely match.")

    return generated_only, reference_only


# --------------------------------------------------
# REFERENCE FIELD PROFILES
# --------------------------------------------------

def profile_reference_fields(answer_key):
    print("\n--- ANSWER KEY VALUE PROFILES ---")

    fields = [
        "promptly",
        "weekend",
        "nwh",
        "high_cash",
        "label",
        "marking",
        "top_n",
    ]

    for field in fields:
        print(f"\n{field}:")
        print(
            answer_key[field]
            .value_counts(dropna=False)
            .sort_index()
            .to_string()
        )


# --------------------------------------------------
# PREPARE GENERATED COMPARISON FIELDS
# --------------------------------------------------

def prepare_generated_fields(generated):
    result = generated.copy()

    entered_dates = pd.to_datetime(
        result["entered_date"],
        errors="coerce",
    )

    if entered_dates.isna().any():
        raise ValueError(
            "Cannot compare weekend: entered_date contains "
            "missing or invalid values."
        )

    # Reference weekend categories:
    # 0 = weekday
    # 1 = Saturday
    # 2 = Sunday

    day_of_week = entered_dates.dt.day_of_week

    result["comparison_weekend"] = day_of_week.map({
        0: 0,
        1: 0,
        2: 0,
        3: 0,
        4: 0,
        5: 1,
        6: 2,
    })


    # Reference nwh matches after_hours directly.
    result["comparison_nwh"] = result["after_hours"].astype(int)

    # Generated metadata fields for answer-key comparison.
    result["comparison_promptly"] = result["promptly"]
    result["comparison_high_cash"] = result["high_cash"].astype(int)    

    return result


# --------------------------------------------------
# FIELD COMPARISON
# --------------------------------------------------

def compare_field(
    merged,
    reference_field,
    generated_field,
    mismatch_records,
):
    generated_values = pd.to_numeric(
        merged[generated_field],
        errors="coerce",
    )

    reference_values = pd.to_numeric(
        merged[reference_field],
        errors="coerce",
    )

    invalid_values = (
        generated_values.isna()
        | reference_values.isna()
    )

    if invalid_values.any():
        print(
            f"\n{reference_field}: "
            f"NOT FULLY COMPARED - "
            f"{invalid_values.sum():,} invalid or missing values."
        )

    comparable = ~invalid_values

    mismatches = (
        comparable
        & (generated_values != reference_values)
    )

    matches = comparable & ~mismatches

    compared_count = int(comparable.sum())
    mismatch_count = int(mismatches.sum())
    match_count = int(matches.sum())

    agreement = (
        match_count / compared_count * 100
        if compared_count > 0
        else 0
    )

    print(f"\n{reference_field}")
    print(f"Compared: {compared_count:,}")
    print(f"Matches: {match_count:,}")
    print(f"Mismatches: {mismatch_count:,}")
    print(f"Agreement: {agreement:.2f}%")

    mismatch_rows = merged.loc[mismatches]

    for _, row in mismatch_rows.iterrows():
        mismatch_records.append({
            "source_row_id": row["source_row_id"],
            "posting_id": row["posting_id_generated"],
            "field": reference_field,
            "generated_value": generated_values.loc[row.name],
            "reference_value": reference_values.loc[row.name],
        })
def diagnose_nwh(merged):
    print("\n--- NWH DEFINITION DIAGNOSTIC ---")

    reference = merged["nwh"].astype(int)

    candidates = {
        "Weekday after-hours only": (
            merged["after_hours"].astype(int)
        ),
        "After-hours including weekends": (
            merged["comparison_nwh"].astype(int)
        ),
    }

    for name, candidate in candidates.items():
        print(f"\n{name}")

        comparison = pd.crosstab(
            candidate,
            reference,
            rownames=["Generated"],
            colnames=["Reference"],
            dropna=False,
        )

        print(comparison)

        matches = (candidate == reference).sum()
        mismatches = (candidate != reference).sum()

        print(f"Matches: {matches:,}")
        print(f"Mismatches: {mismatches:,}")
        print(f"Agreement: {matches / len(merged) * 100:.2f}%")
def inspect_nwh_mismatches(merged):
    print("\n--- NWH MISMATCH DETAILS ---")

    generated = merged["after_hours"].astype(int)
    reference = merged["nwh"].astype(int)

    mismatches = merged.loc[generated != reference].copy()

    mismatches["entered_date"] = pd.to_datetime(
        mismatches["entered_date"]
    )

    mismatches["hour"] = mismatches["entered_date"].dt.hour
    mismatches["minute"] = mismatches["entered_date"].dt.minute
    mismatches["weekday"] = mismatches["entered_date"].dt.day_name()

    print("\nMismatch counts by classification:")
    print(pd.crosstab(generated.loc[mismatches.index],
                      reference.loc[mismatches.index]))

    print("\nMismatch counts by hour:")
    print(
        mismatches.groupby(
            ["after_hours", "nwh", "hour"]
        ).size().to_string()
    )

    print("\nMismatch counts by weekday:")
    print(
        mismatches.groupby(
            ["after_hours", "nwh", "weekday"]
        ).size().to_string()
    )

    print("\nSample mismatches:")
    print(
        mismatches[
            [
                "source_row_id",
                "posting_id_generated",
                "entered_date",
                "after_hours",
                "nwh",
            ]
        ].head(30).to_string(index=False)
    )
        

# --------------------------------------------------
# MAIN RECONCILIATION
# --------------------------------------------------

def main():
    print("Loading generated metadata and answer key...")

    generated, answer_key = load_comparison_files()

    generated_only, reference_only = validate_structure(
        generated,
        answer_key,
    )

    profile_reference_fields(answer_key)

    generated = prepare_generated_fields(generated)

    merged = generated.merge(
        answer_key,
        on="source_row_id",
        how="inner",
        suffixes=("_generated", "_reference"),
        validate="one_to_one",
    )

    # Restore clean reference field names after merge.
    merged["promptly"] = merged["promptly_reference"]
    merged["high_cash"] = merged["high_cash_reference"]

    diagnose_nwh(merged)
    inspect_nwh_mismatches(merged)

    print("\n--- POSTING ID VALIDATION ---")

    posting_id_mismatches = (
        merged["posting_id_generated"]
        != merged["posting_id_reference"]
    )

    print(
        "Posting ID mismatches: "
        f"{posting_id_mismatches.sum():,}"
    )

    mismatch_records = []

    for _, row in merged.loc[posting_id_mismatches].iterrows():
        mismatch_records.append({
            "source_row_id": row["source_row_id"],
            "posting_id": (
                f'{row["posting_id_generated"]} != '
                f'{row["posting_id_reference"]}'
            ),
            "field": "posting_id",
            "generated_value": row["posting_id_generated"],
            "reference_value": row["posting_id_reference"],
        })

    print("\n--- METADATA RECONCILIATION ---")

    compare_field(
        merged,
        "weekend",
        "comparison_weekend",
        mismatch_records,
    )

    compare_field(
        merged,
        "nwh",
        "comparison_nwh",
        mismatch_records,
    )

    compare_field(
        merged,
        "promptly",
        "comparison_promptly",
        mismatch_records,
    )

    compare_field(
        merged,
        "high_cash",
        "comparison_high_cash",
        mismatch_records,
    )

    print("\n--- REFERENCE ONLY FIELDS ---")
    print("label: Reference-only anomaly label.")
    print("marking: Reference-only anomaly scenario identifier.")
    print("top_n: Reference-only audit-scope indicator.")

    # Export every mismatch.

    mismatch_df = pd.DataFrame(
        mismatch_records,
        columns=[
            "source_row_id",
            "posting_id",
            "field",
            "generated_value",
            "reference_value",
        ],
    )

    mismatch_df.to_csv(
        MISMATCH_OUTPUT_FILE,
        sep=";",
        index=False,
    )

    print("\n--- FINAL RESULT ---")
    print(f"Rows compared: {len(merged):,}")
    print(f"Total mismatches: {len(mismatch_df):,}")
    print(f"Mismatch report: {MISMATCH_OUTPUT_FILE}")

    if generated_only or reference_only:
        print(
            "WARNING: Some source rows were excluded from "
            "field comparisons because they did not have a "
            "matching source_row_id."
        )

    if mismatch_df.empty:
        print("No mismatches found in the compared fields.")
    else:
        print("\nFirst 20 mismatches:")
        print(mismatch_df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()


