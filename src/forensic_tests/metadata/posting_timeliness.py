import pandas as pd


def run_test(df: pd.DataFrame) -> pd.DataFrame:
    required_columns = [
        "source_row_id",
        "document_date",
        "entered_date",
    ]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    document_dates = pd.to_datetime(
        df["document_date"],
        errors="coerce"
    )

    entered_dates = pd.to_datetime(
        df["entered_date"],
        errors="coerce"
    )

    if document_dates.isna().any() or entered_dates.isna().any():
        raise ValueError(
            "Cannot calculate posting timeliness: "
            "document_date or entered_date contains missing "
            "or invalid values."
        )

    posting_delay_days = (
        entered_dates - document_dates
    ).dt.total_seconds() / 86400

    return pd.DataFrame({
        "source_row_id": df["source_row_id"],
        "posting_delay_days": posting_delay_days,
    })