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



for col in ["promptly", "weekend", "nwh", "marking"]:
    print(f"\n{col}")
    print(df[col].value_counts().sort_index())