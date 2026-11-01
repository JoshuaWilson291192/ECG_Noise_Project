from pathlib import Path
import ast
import pandas as pd
import torch
import wfdb
from torch.utils.data import Dataset, DataLoader

FOLDER = Path(__file__).resolve().parent
DATA = FOLDER / "data" / "ptb-xl"
RESULTS = FOLDER / "results"

# Five ECG classes.
CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]

#Expected recording format
EXPECTED_FS = 500
EXPECTED_SHAPE = (5000, 12) 

class ECGDataset(Dataset):
    def __init__(self, records):
        self.records = records.reset_index()

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        row = self.records.iloc[i]

        # Load one ECG file
        signal, info = wfdb.rdsamp(str(DATA / row["filename_hr"]))
        # Check format is as expected (500 Hz, 12 leads, 10 seconds)
        if info["fs"] != EXPECTED_FS or signal.shape != EXPECTED_SHAPE:
            raise ValueError(f"Unexpected ECG format for ID {row['ecg_id']}")

        x = torch.tensor(signal.T, dtype=torch.float32)  # the ECG signal
        y = torch.tensor([float(c in row["labels"]) for c in CLASSES])  # 1 = has condition, 0 = does not
        return x, y, int(row["ecg_id"])


def create_dataloaders(batch_size=16):
    # Read the recording details and label list.
    df = pd.read_csv(DATA / "ptbxl_database.csv", index_col="ecg_id")
    codes = pd.read_csv(DATA / "scp_statements.csv", index_col=0)
    class_of_code = codes.loc[codes["diagnostic"] == 1, "diagnostic_class"].dropna().to_dict()

    def get_labels(text):
        found = {class_of_code.get(code) for code in ast.literal_eval(text)}
        return [c for c in CLASSES if c in found]

    # Add a label list and filter out recordings with no labels
    df["labels"] = df["scp_codes"].apply(get_labels)
    df = df[df["labels"].str.len() > 0]

    # Split into training, validation and test sets using the dataset's own fold numbers.
    splits = {
        "train": df[df["strat_fold"].between(1, 8)],
        "validation": df[df["strat_fold"] == 9],
        "test": df[df["strat_fold"] == 10],
    }

    # Make sure patients do not appear in multiple splits.
    if sum(s["patient_id"].nunique() for s in splits.values()) != df["patient_id"].nunique():
        raise ValueError("Patient overlap detected between dataset splits.")

    # Save split details and counts.
    RESULTS.mkdir(parents=True, exist_ok=True)
    summary = []
    for name, split in splits.items():
        split[["patient_id", "strat_fold", "filename_hr", "labels"]].to_csv(
            RESULTS / f"{name}_recordings.csv", index_label="ecg_id"
        )
        counts = {"split": name, "recordings": len(split), "patients": split["patient_id"].nunique()}
        for c in CLASSES:
            counts[c] = sum(c in labels for labels in split["labels"])
        summary.append(counts)
    pd.DataFrame(summary).to_csv(RESULTS / "dataset_summary.csv", index=False)

    # Shuffle training data only.
    return tuple(
        DataLoader(ECGDataset(s), batch_size=batch_size, shuffle=(name == "train"))
        for name, s in splits.items()
    )


# Test if working
if __name__ == "__main__":
    train_loader, val_loader, test_loader = create_dataloaders(batch_size=4)
    x, y, ids = next(iter(train_loader))
    assert x.shape[1:] == (12, 5000) and y.shape[1:] == (5,)
    print("Data loader is working correctly.")
    print("ECG batch shape:", x.shape)
    print("Label batch shape:", y.shape)
    print("First ECG ID:", ids[0].item())