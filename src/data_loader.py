
from pathlib import Path
import pandas as pd
import numpy as np


# --------------------------------------------------
# 1. File location and expected schema
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATASET_PATH = (
    PROJECT_ROOT
    / "datasets"
    / "raw"
    / "journal_entries_raw.csv"
)

REQUIRED_COLUMNS = [
    "source_row_id",
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


# --------------------------------------------------
# 2. Data normalization
# --------------------------------------------------

def normalize_dates(df):

    date_formats = {
        "document_date": "%Y-%m-%d",
        "posting_date": "%Y-%m-%d",
        "entered_date": "%Y-%m-%d %H:%M:%S",
    }

    for column, date_format in date_formats.items():

        original = df[column]

        converted = pd.to_datetime(
            original,
            format=date_format,
            errors="coerce"
        )

        invalid = original.notna() & converted.isna()

        if invalid.any():
            bad_rows = df.loc[
                invalid, "source_row_id"
            ].head(10).tolist()

            raise ValueError(
                f"Invalid dates in '{column}'. "
                f"Example source_row_id values: {bad_rows}"
            )

        df[column] = converted

    return df


def normalize_numeric_columns(df):

    numeric_columns = [
        "amount",
        "tax_rate",
    ]

    for column in numeric_columns:

        original = df[column]

        converted = pd.to_numeric(
            original,
            errors="coerce"
        )

        invalid = original.notna() & converted.isna()

        if invalid.any():
            bad_rows = df.loc[
                invalid, "source_row_id"
            ].head(10).tolist()

            raise ValueError(
                f"Invalid numeric values in '{column}'. "
                f"Example source_row_id values: {bad_rows}"
            )

        df[column] = converted

    return df


def normalize_identifiers(df):

    string_columns = [
        "posting_id",
        "gl_account",
        "gl_account_name",
        "cd_flag",
        "user",
        "ref_id",
        "text",
    ]

    for column in string_columns:
        df[column] = df[column].astype("string")

    # Remove surrounding whitespace from identifiers
    # and categorical fields.

    for column in [
        "posting_id",
        "gl_account",
        "gl_account_name",
        "cd_flag",
        "user",
        "ref_id",
    ]:
        df[column] = df[column].str.strip()

    # Standardize debit/credit indicators.

    df["cd_flag"] = df["cd_flag"].str.upper()

    return df


# --------------------------------------------------
# 3. Dataset loading
# --------------------------------------------------

def load_dataset(file_path=RAW_DATASET_PATH):

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

    # Read identifier fields as strings to avoid
    # losing formatting such as leading zeros.

    df = pd.read_csv(
        file_path,
        sep=";",
        dtype={
            "posting_id": "string",
            "gl_account": "string",
            "gl_account_name": "string",
            "cd_flag": "string",
            "user": "string",
            "ref_id": "string",
            "text": "string",
        }
    )

    # Check the expected schema.

    missing_columns = [
        column for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: "
            f"{missing_columns}"
        )

    # Reorder columns consistently.

    df = df[REQUIRED_COLUMNS].copy()

    # Normalize representations.

    df = normalize_identifiers(df)
    df = normalize_dates(df)
    df = normalize_numeric_columns(df)

    validation_report = validate_dataset(df)

    if not validation_report["valid"]:
        raise ValueError(
            f"Dataset validation failed: "
            f"{validation_report['errors']}"
        )

    return df


# --------------------------------------------------
# 3. Row-level integrity validation
# --------------------------------------------------

def validate_dataset(df):

    errors = []
    warnings = []

    # --------------------------------------------------
    # A. Identifier validation
    # --------------------------------------------------

    # source_row_id must be present and unique.

    missing_ids = df["source_row_id"].isna()

    if missing_ids.any():
        errors.append({
            "check": "Missing source_row_id",
            "count": int(missing_ids.sum()),
            "message": "Some posting lines have no source identifier."
        })

    duplicate_ids = df["source_row_id"].duplicated(keep=False)

    if duplicate_ids.any():
        errors.append({
            "check": "Duplicate source_row_id",
            "count": int(duplicate_ids.sum()),
            "message": "Some posting lines share a source identifier."
        })

    # posting_id must exist because it identifies
    # the complete journal entry.

    missing_posting_ids = (
        df["posting_id"].isna()
        | df["posting_id"].str.strip().eq("")
    )

    if missing_posting_ids.any():
        errors.append({
            "check": "Missing posting_id",
            "count": int(missing_posting_ids.sum()),
            "message": "Some posting lines cannot be assigned to a journal entry."
        })


    # --------------------------------------------------
    # B. Numeric validation
    # --------------------------------------------------

    # Amounts must be numeric and finite.

    invalid_amounts = (
        df["amount"].isna()
        | ~np.isfinite(df["amount"])
    )

    if invalid_amounts.any():
        errors.append({
            "check": "Invalid amount",
            "count": int(invalid_amounts.sum()),
            "message": "Some amounts are missing or non-finite."
        })

    # Tax rate must be numeric and finite when present.

    invalid_tax_rates = (
        df["tax_rate"].notna()
        & ~np.isfinite(df["tax_rate"])
    )

    if invalid_tax_rates.any():
        errors.append({
            "check": "Invalid tax rate",
            "count": int(invalid_tax_rates.sum()),
            "message": "Some tax rates are non-finite."
        })


    # --------------------------------------------------
    # C. Debit / credit validation
    # --------------------------------------------------

    invalid_flags = (
        df["cd_flag"].isna()
        | ~df["cd_flag"].isin(["D", "C"])
    )

    if invalid_flags.any():
        errors.append({
            "check": "Invalid debit/credit indicator",
            "count": int(invalid_flags.sum()),
            "message": "Debit/credit indicators must be D or C."
        })


    # --------------------------------------------------
    # D. Missing descriptive information
    # --------------------------------------------------

    descriptive_columns = [
        "gl_account_name",
        "user",
        "text",
    ]

    for column in descriptive_columns:

        missing_values = (
            df[column].isna()
            | df[column].str.strip().eq("")
        )

        if missing_values.any():
            warnings.append({
                "check": f"Missing {column}",
                "count": int(missing_values.sum()),
                "message": f"Some records have no {column}."
            })


    # --------------------------------------------------
    # E. Return validation report
    # --------------------------------------------------

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
    }


# --------------------------------------------------
# 4. Journal-entry grouping
# --------------------------------------------------


def group_journal_entries(df):

    return df.groupby(
        "posting_id",
        sort=False
    )

