import pandas as pd

DEFAULT_THRESHOLD = 1500.00


def run_test(
    df: pd.DataFrame,
    threshold: float = DEFAULT_THRESHOLD
) -> pd.DataFrame:
    """
    Identifies journal entries involving cash where the
    posting amount exceeds the configured materiality threshold.

    If an entry qualifies, every posting line in that entry
    receives True.

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

    amounts = pd.to_numeric(
        df["amount"],
        errors="coerce"
    )

    if amounts.isna().any() or not amounts.map(
        lambda value: abs(value) != float("inf")
    ).all():
        raise ValueError(
            "Cannot determine high-cash activity: "
            "amount contains missing or invalid values."
        )

    if df["involves_cash"].isna().any():
        raise ValueError(
            "Cannot determine high-cash activity: "
            "involves_cash contains missing values."
        )

    # Identify qualifying posting lines.
    qualifying_lines = (
        df["involves_cash"].astype(bool)
        & (amounts.abs() > threshold)
    )

    # Propagate a qualifying result to every line in the entry.
    high_cash_entries = (
        pd.DataFrame({
            "posting_id": df["posting_id"],
            "_qualifies": qualifying_lines
        })
        .groupby("posting_id", sort=False)["_qualifies"]
        .transform("any")
    )

    return pd.DataFrame({
        "source_row_id": df["source_row_id"],
        "high_cash": high_cash_entries
    })