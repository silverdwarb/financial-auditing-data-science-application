
import pandas as pd


def run_test(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identifies journal entries involving cash.

    If any posting line within a journal entry has 'cash' in its
    gl_account_name, every posting line belonging to that journal
    entry receives involves_cash = True.

    Returns:
        DataFrame containing source_row_id and involves_cash.
    """

    required_columns = [
        "source_row_id",
        "posting_id",
        "gl_account_name",
    ]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # Identify posting lines whose account name contains "cash".
    cash_lines = (
        df["gl_account_name"]
        .fillna("")
        .str.contains("cash", case=False, regex=False)
    )

    # Identify every journal entry containing at least one cash line.
    cash_entries = (
        df.assign(_cash_line=cash_lines)
        .groupby("posting_id")["_cash_line"]
        .any()
    )

    # Assign the journal-entry-level result to every posting line.
    results = pd.DataFrame({
        "source_row_id": df["source_row_id"],
        "involves_cash": df["posting_id"].map(cash_entries)
    })

    return results
