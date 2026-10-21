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

# Mapping diagnostic codes to sublcasses
mapping = scp.loc[scp["diagnostic"] == 1,"diagnostic_class"].dropna().to_dict()

def get_labels(codes):
    return sorted({
        mapping[code]
        for code in codes
        if code in mapping
    })

db["labels"] = db["scp_codes"].apply(get_labels)
print(db[["ecg_id", "labels"]].head())

# Plot lead test, single and 12
import wfdb
import numpy as np
import matplotlib.pyplot as plt

row = db.iloc[0]
record_path = DATA / row["filename_hr"]
record = wfdb.rdrecord(str(record_path))
signal = record.p_signal
fs = record.fs

print("Signal shape:", signal.shape)
print("Sampling frequency:", fs)
print("Leads:", record.sig_name)

time = np.arange(signal.shape[0]) / fs

#first lead 
plt.figure(figsize=(12, 4))
plt.plot(time, signal[:, 0])
plt.title("ECG - Lead I")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude (mV)")
plt.grid(True)
plt.show()

#12 leads
fig, axes = plt.subplots(
    12, 1, figsize=(12, 18), sharex=True
)

for i, ax in enumerate(axes):
    ax.plot(time, signal[:, i])
    ax.set_ylabel(record.sig_name[i])
    ax.grid(True)

axes[-1].set_xlabel("Time (seconds)")
plt.tight_layout()
plt.show()
