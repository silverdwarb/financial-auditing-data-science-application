
import numpy as np
import pandas as pd


def run_test(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identifies posting-line amounts that are multiples of
    10, 100, or 1,000.

    Negative amounts are evaluated by their absolute value.
    Zero amounts are excluded.

    Returns:
        DataFrame containing source_row_id and round-number flags.
    """

    required_columns = [
        "source_row_id",
        "amount",
    ]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    amounts = pd.to_numeric(
        df["amount"],
        errors="coerce"
    )

    if amounts.isna().any() or not np.isfinite(amounts).all():
        raise ValueError(
            "Cannot evaluate round numbers: "
            "amount contains missing, invalid, or infinite values."
        )

    absolute_amounts = amounts.abs()

    results = pd.DataFrame({
        "source_row_id": df["source_row_id"]
    })

    for interval in [10, 100, 1000]:
        # Check whether the amount is approximately an exact multiple.
        quotient = absolute_amounts / interval

        results[f"round_{interval}"] = (
            (absolute_amounts != 0)
            & np.isclose(
                quotient,
                np.round(quotient),
                rtol=0,
                atol=1e-9
            )
        )

    return results
