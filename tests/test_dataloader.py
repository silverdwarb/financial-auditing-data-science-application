import pandas as pd
import pytest

from src.data_loader import (
    load_dataset,
    validate_dataset,
    group_journal_entries,
)


# --------------------------------------------------
# Test fixture: small, valid dataset
# --------------------------------------------------

@pytest.fixture
def sample_dataset():
    """Create a small dataset containing two journal entries."""

    return pd.DataFrame({
        "source_row_id": [0, 1, 2, 3],
        "posting_id": ["A", "A", "B", "B"],
        "document_date": pd.to_datetime([
            "2026-01-01",
            "2026-01-01",
            "2026-01-02",
            "2026-01-02",
        ]),
        "posting_date": pd.to_datetime([
            "2026-01-01",
            "2026-01-01",
            "2026-01-02",
            "2026-01-02",
        ]),
        "entered_date": pd.to_datetime([
            "2026-01-01 10:00:00",
            "2026-01-01 10:00:00",
            "2026-01-02 11:00:00",
            "2026-01-02 11:00:00",
        ]),
        "gl_account": pd.Series(
            ["1000", "2000", "1000", "3000"],
            dtype="string",
        ),
        "gl_account_name": pd.Series(
            ["Cash", "Revenue", "Cash", "Expense"],
            dtype="string",
        ),
        "amount": [100.0, 100.0, 200.0, 200.0],
        "cd_flag": pd.Series(
            ["D", "C", "D", "C"],
            dtype="string",
        ),
        "user": pd.Series(
            ["Alice", "Alice", "Bob", "Bob"],
            dtype="string",
        ),
        "tax_rate": [0.0, 0.0, 0.1, 0.1],
        "ref_id": pd.Series(
            ["R1", "R1", "R2", "R2"],
            dtype="string",
        ),
        "text": pd.Series(
            ["Sale", "Sale", "Purchase", "Purchase"],
            dtype="string",
        ),
    })


# --------------------------------------------------
# 1. Dataset loading
# --------------------------------------------------

def test_load_dataset_returns_dataframe():
    """The real raw dataset should load successfully."""

    df = load_dataset()

    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0


def test_load_dataset_has_expected_columns():
    """The loader should return the expected schema."""

    df = load_dataset()

    expected_columns = [
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

    assert df.columns.tolist() == expected_columns


# --------------------------------------------------
# 2. Data integrity validation
# --------------------------------------------------

def test_valid_dataset_passes_validation(sample_dataset):
    """A correctly structured dataset should pass validation."""

    report = validate_dataset(sample_dataset)

    assert report["valid"] is True
    assert report["errors"] == []


def test_duplicate_source_row_ids_are_detected(sample_dataset):
    """Duplicate source identifiers should produce an error."""

    sample_dataset.loc[1, "source_row_id"] = 0

    report = validate_dataset(sample_dataset)

    assert report["valid"] is False
    assert any(
        error["check"] == "Duplicate source_row_id"
        for error in report["errors"]
    )


def test_missing_posting_id_is_detected(sample_dataset):
    """A posting line without a journal-entry ID is invalid."""

    sample_dataset.loc[0, "posting_id"] = pd.NA

    report = validate_dataset(sample_dataset)

    assert report["valid"] is False
    assert any(
        error["check"] == "Missing posting_id"
        for error in report["errors"]
    )


def test_invalid_debit_credit_indicator_is_detected(sample_dataset):
    """Only D and C are accepted as debit/credit indicators."""

    sample_dataset.loc[0, "cd_flag"] = "X"

    report = validate_dataset(sample_dataset)

    assert report["valid"] is False
    assert any(
        error["check"] == "Invalid debit/credit indicator"
        for error in report["errors"]
    )


def test_missing_amount_is_detected(sample_dataset):
    """An amount required for analysis cannot be missing."""

    sample_dataset.loc[0, "amount"] = float("nan")

    report = validate_dataset(sample_dataset)

    assert report["valid"] is False
    assert any(
        error["check"] == "Invalid amount"
        for error in report["errors"]
    )


def test_non_finite_amount_is_detected(sample_dataset):
    """Infinite amounts must not pass numeric validation."""

    sample_dataset.loc[0, "amount"] = float("inf")

    report = validate_dataset(sample_dataset)

    assert report["valid"] is False
    assert any(
        error["check"] == "Invalid amount"
        for error in report["errors"]
    )


# --------------------------------------------------
# 3. Journal-entry grouping
# --------------------------------------------------

def test_grouping_finds_correct_number_of_entries(sample_dataset):
    """Two distinct posting IDs should produce two groups."""

    entries = group_journal_entries(sample_dataset)

    assert entries.ngroups == 2


def test_grouping_preserves_posting_lines(sample_dataset):
    """Each group should contain its original posting lines."""

    entries = group_journal_entries(sample_dataset)

    entry_a = entries.get_group("A")
    entry_b = entries.get_group("B")

    assert len(entry_a) == 2
    assert len(entry_b) == 2

    assert entry_a["source_row_id"].tolist() == [0, 1]
    assert entry_b["source_row_id"].tolist() == [2, 3]


def test_grouping_preserves_all_rows(sample_dataset):
    """Grouping should not lose or duplicate posting lines."""

    entries = group_journal_entries(sample_dataset)

    grouped_row_count = sum(
        len(entry)
        for _, entry in entries
    )

    assert grouped_row_count == len(sample_dataset)