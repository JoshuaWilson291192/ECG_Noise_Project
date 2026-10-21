from pathlib import Path
import pandas as pd
import ast 

DATA = Path("data/ptb-xl")

# Load and print the database and statements
db = pd.read_csv(DATA / "ptbxl_database.csv")
scp = pd.read_csv(DATA / "scp_statements.csv", index_col=0)

print("Number of ECG records:", len(db)) # 21799 records
print(db[["ecg_id", "scp_codes", "filename_hr"]].head())

# Convert from string to list
db["scp_codes"] = db["scp_codes"].apply(ast.literal_eval)

# Mapping diagnostic codes to
mapping = scp.loc[scp["diagnostic"] == 1,"diagnostic_class"].dropna().to_dict()

def get_labels(codes):
    return sorted({
        mapping[code]
        for code in codes
        if code in mapping
    })

db["labels"] = db["scp_codes"].apply(get_labels)
print(db[["ecg_id", "labels"]].head())
