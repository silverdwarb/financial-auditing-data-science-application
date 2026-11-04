
import pandas as pd

from src.forensic_tests.metadata import (
    involves_cash,
    high_cash,
    weekend_activity,
    after_hours,
    posting_timeliness,
    round_numbers,
)


def add_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """
    Runs all metadata modules and merges their outputs
    into the original journal-entry posting-line dataset.

    Every module must return source_row_id and its
    corresponding metadata columns.

    Returns:
        DataFrame containing original data and all generated metadata.
    """

    if "source_row_id" not in df.columns:
        raise ValueError(
            "Input DataFrame must contain source_row_id."
        )

    # Preserve the original dataset.
    result = df.copy()

    # Metadata modules are executed in dependency order.
    metadata_modules = [
        involves_cash,
        high_cash,
        weekend_activity,
        after_hours,
        posting_timeliness,
        round_numbers,
    ]

    for module in metadata_modules:

        # High cash requires involves_cash to be available.
        if module == high_cash:
            module_input = result
        else:
            module_input = result

        metadata = module.run_test(module_input)

        # Validate module output.
        if "source_row_id" not in metadata.columns:
            raise ValueError(
                f"{module.__name__} did not return source_row_id."
            )

        if metadata["source_row_id"].duplicated().any():
            raise ValueError(
                f"{module.__name__} returned duplicate source_row_id values."
            )

        if set(metadata["source_row_id"]) != set(result["source_row_id"]):
            raise ValueError(
                f"{module.__name__} returned mismatched source_row_id values."
            )

        # Prevent accidental overwriting of existing columns.
        new_columns = [
            column for column in metadata.columns
            if column != "source_row_id"
        ]

        overlapping_columns = [
            column for column in new_columns
            if column in result.columns
        ]

        if overlapping_columns:
            raise ValueError(
                f"{module.__name__} attempted to overwrite existing "
                f"columns: {overlapping_columns}"
            )

        # Merge module output into consolidated dataset.
        result = result.merge(
            metadata,
            on="source_row_id",
            how="left",
            validate="one_to_one",
            sort=False,
        )

    return result
