import pandas as pd

df = pd.read_csv("datasets/SOT/journal_entries_4203.csv", sep=";")

# print(df.shape)
# print(df.head(10).to_string())
# print(df.dtypes)

# print("Number of columns:", len(df.columns))
# print("Column names:")
# print(df.columns.tolist())

# print("Unique journal entries:", df["posting_id"].nunique())
# print("Unique GL accounts:", df["gl_account"].nunique())
# print("Unique users:", df["user"].nunique())

# for col in ["label", "marking", "top_n", "high_cash"]:
#     print(f"\n{col}:")
#     print(df[col].value_counts(dropna=False))



# for col in ["promptly", "weekend", "nwh", "marking"]:
#     print(f"\n{col}")
#     print(df[col].value_counts().sort_index())

# import pandas as pd

# df = pd.read_csv(
#     "datasets/raw/journal_entries_raw.csv",
#     sep=";"
# )

# print("DATA TYPES:")
# print(df.dtypes)

# print("\nSAMPLE VALUES:")
# for column in [
#     "document_date",
#     "posting_date",
#     "entered_date",
#     "amount",
#     "gl_account",
#     "cd_flag",
#     "tax_rate",
#     "user",
# ]:
#     print(f"\n{column}:")
#     print(df[column].head(5).tolist())

# print("\nMISSING VALUES:")
# print(df.isna().sum())

import sys
import os
# 1. Get the directory of the current script (scripts/)
current_dir = os.path.dirname(os.path.abspath(__file__))
# 2. Get the parent directory (financial-auditing-data-science-application/)
parent_dir = os.path.dirname(current_dir)
# 3. Add the parent directory to Python's search path
sys.path.append(parent_dir)

from  src.data_loader import load_dataset, group_journal_entries; 
df = load_dataset(); 
entries = group_journal_entries(df)
print('Total journal entries:', len(entries))
print('\nExample journal entry:')
print(entries.get_group("a_1"))