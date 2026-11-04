
import pandas as pd

from src.forensic_tests.add_metadata import add_metadata

def sample_dataset():
    """
    Small dataset for testing metadata aggregation.
    Includes cash/non-cash entries and weekday/weekend timestamps.
    """

    return pd.DataFrame({
        "source_row_id": [0, 1, 2, 3, 4],
        "posting_id": ["a_1", "a_1", "a_2", "a_2", "a_3"],
        "document_date": pd.to_datetime([
            "2026-10-26",
            "2026-10-26",
            "2026-10-27",
            "2026-10-27",
            "2026-10-28",
        ]),
        "posting_date": pd.to_datetime([
            "2026-10-26",
            "2026-10-26",
            "2026-10-27",
            "2026-10-27",
            "2026-10-28",
        ]),
        "entered_date": pd.to_datetime([
            "2026-10-26 10:00",
            "2026-10-26 10:00",
            "2026-10-31 12:00",
            "2026-10-31 12:00",
            "2026-10-28 19:00",
        ]),
        "gl_account": ["1000", "2000", "3000", "1000", "4000"],
        "gl_account_name": [
            "Cash-in-hand",
            "Accounts Payable",
            "Revenue",
            "Cash-in-hand",
            "Expenses",
        ],
        "amount": [2000.00, 2000.00, 500.00, 500.00, 1500.00],
        "cd_flag": ["D", "C", "C", "D", "D"],
        "user": ["David"] * 5,
        "tax_rate": [0.0] * 5,
        "ref_id": [""] * 5,
        "text": ["Test entry"] * 5,
    })


def test_aggregator_adds_all_metadata():
    df = sample_dataset()

    result = add_metadata(df)

    expected_columns = [
        "involves_cash",
        "high_cash",
        "weekend_activity",
        "after_hours",
        "posting_delay_days",
        "round_10",
        "round_100",
        "round_1000",
    ]

    for column in expected_columns:
        assert column in result.columns


def test_aggregator_preserves_all_rows():
    df = sample_dataset()

    result = add_metadata(df)

    assert len(result) == len(df)
    assert result["source_row_id"].is_unique


def test_cash_metadata_propagates_across_entry():
    df = sample_dataset()

    result = add_metadata(df)

    entry = result[result["posting_id"] == "a_1"]

    assert entry["involves_cash"].all()


def test_non_cash_entry_is_false():
    df = sample_dataset()

    result = add_metadata(df)

    entry = result[result["posting_id"] == "a_3"]

    assert not entry["involves_cash"].any()


def test_high_cash_uses_cash_metadata():
    df = sample_dataset()

    result = add_metadata(df)

    cash_entry = result[result["posting_id"] == "a_1"]

    assert cash_entry["high_cash"].all()

    non_cash_entry = result[result["posting_id"] == "a_3"]

    assert not non_cash_entry["high_cash"].any()


def test_weekend_metadata_propagates():
    df = sample_dataset()

    result = add_metadata(df)

    weekend_entry = result[result["posting_id"] == "a_2"]

    assert weekend_entry["weekend_activity"].all()


def test_after_hours_metadata():
    df = sample_dataset()

    result = add_metadata(df)

    after_hours_entry = result[result["posting_id"] == "a_3"]

    assert after_hours_entry["after_hours"].all()


def test_original_columns_are_preserved():
    df = sample_dataset()

    result = add_metadata(df)

    for column in df.columns:
        assert column in result.columns
        pd.testing.assert_series_equal(
            result[column],
            df[column]
        )
