
import pandas as pd
from datetime import time


def run_test(
    df: pd.DataFrame,
    business_start: str = "08:00",
    business_end: str = "18:00"
) -> pd.DataFrame:
    """
    Identifies journal entries containing posting lines entered
    outside configured business hours, regardless of weekday.

    If any line in a journal entry qualifies, every line in that
    entry receives True.

    Returns:
        DataFrame containing source_row_id and after_hours.
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

    try:
        start = time.fromisoformat(business_start)
        end = time.fromisoformat(business_end)
    except ValueError:
        raise ValueError(
            "Business hours must use HH:MM format."
        )

    if start >= end:
        raise ValueError(
            "Business start must be earlier than business end."
        )

    entered_dates = pd.to_datetime(
        df["entered_date"],
        errors="coerce"
    )

    if entered_dates.isna().any():
        raise ValueError(
            "Cannot determine after-hours activity: "
            "entered_date contains missing or invalid values."
        )

    # Identify entries outside configured business hours.
    outside_hours = (
        (entered_dates.dt.time < start)
        | (entered_dates.dt.time >= end)
    )

    working_data = pd.DataFrame({
        "posting_id": df["posting_id"],
        "_after_hours": outside_hours
    })

    after_hours_entries = (
        working_data
        .groupby("posting_id", sort=False)["_after_hours"]
        .transform("any")
    )

    return pd.DataFrame({
        "source_row_id": df["source_row_id"],
        "after_hours": after_hours_entries
    })
