
import pandas as pd

DEFAULT_THRESHOLD = 1500.00


def run_test(
    df: pd.DataFrame,
    threshold: float = DEFAULT_THRESHOLD
) -> pd.DataFrame:
    """
    Identifies cash-related journal entries whose individual
    posting lines exceed the configured amount threshold.

    A journal entry must involve cash to qualify.

    Returns:
        DataFrame containing source_row_id and high_cash.
    """

    required_columns = [
        "source_row_id",
        "posting_id",
        "amount",
        "involves_cash",
    ]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    if threshold < 0:
        raise ValueError("Threshold cannot be negative.")

    results = pd.DataFrame({
        "source_row_id": df["source_row_id"],
        "high_cash": (
            df["involves_cash"]
            & (df["amount"].abs() > threshold)
        )
    })

    return results
