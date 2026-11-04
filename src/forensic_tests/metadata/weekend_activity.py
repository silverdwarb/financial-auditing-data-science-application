import pandas as pd


def run_test(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identifies journal entries containing posting lines entered
    on Saturday or Sunday.

    If any posting line in a journal entry was entered on a
    weekend, every line in that journal entry receives True.

    Returns:
        DataFrame containing source_row_id and weekend_activity.
    """

    required_columns = [
        "source_row_id",
        "posting_id",
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

    # Convert timestamps and reject missing or invalid dates.
    entered_dates = pd.to_datetime(
        df["entered_date"],
        errors="coerce"
    )

    if entered_dates.isna().any():
        raise ValueError(
            "Cannot determine weekend activity: "
            "entered_date contains missing or invalid values."
        )

    # Saturday = 5, Sunday = 6.
    weekend_lines = entered_dates.dt.day_of_week >= 5

    # Propagate weekend status across the entire journal entry.
    working_data = pd.DataFrame({
        "posting_id": df["posting_id"],
        "_weekend": weekend_lines
    })

    weekend_entries = (
        working_data
        .groupby("posting_id", sort=False)["_weekend"]
        .transform("any")
    )

    return pd.DataFrame({
        "source_row_id": df["source_row_id"],
        "weekend_activity": weekend_entries
    })
