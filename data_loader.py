
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

# Expected recording size.
EXPECTED_FS = 500
EXPECTED_SHAPE = (5000, 12)


class ECGDataset(Dataset):
    def __init__(self, records):
        # Keep ECG IDs as a column.
        self.records = records.reset_index()

    def __len__(self):
        # Tell PyTorch how many ECGs are available.
        return len(self.records)

    def __getitem__(self, i):
        row = self.records.iloc[i]

        # Load one ECG file.
        signal, info = wfdb.rdsamp(str(DATA / row["filename_hr"]))

        # Check the ECG format.
        if info["fs"] != EXPECTED_FS or signal.shape != EXPECTED_SHAPE:
            raise ValueError(f"Unexpected ECG format for ID {row['ecg_id']}")

        # Put leads before samples.
        x = torch.tensor(signal.T.copy(), dtype=torch.float32)

        # Turn labels into 0s and 1s.
        y = torch.tensor(
            [float(label in row["labels"]) for label in CLASSES],
            dtype=torch.float32,
        )

        return x, y, int(row["ecg_id"])


def create_dataloaders(batch_size=16):
    # Read the recording details and label list.
    df = pd.read_csv(DATA / "ptbxl_database.csv", index_col="ecg_id")
    scp = pd.read_csv(DATA / "scp_statements.csv", index_col=0)

    # Match codes to our five classes.
    code_to_class = (
        scp.loc[scp["diagnostic"] == 1, "diagnostic_class"]
        .dropna()
        .to_dict()
    )

    def get_labels(text):
        codes = ast.literal_eval(text)
        mapped_classes = {code_to_class.get(code) for code in codes}
        return [label for label in CLASSES if label in mapped_classes]

    # Convert label text into Python lists.
    df["labels"] = df["scp_codes"].apply(get_labels)

    # Remove ECGs without a target label.
    valid = df["labels"].str.len() > 0
    excluded = (~valid).sum()
    df = df[valid].copy()

    # Use the official train, validation, and test splits.
    splits = {
        "train": df[df["strat_fold"].between(1, 8)],
        "validation": df[df["strat_fold"] == 9],
        "test": df[df["strat_fold"] == 10],
    }

    # Make sure patients do not appear in multiple splits.
    patient_sets = [set(split["patient_id"]) for split in splits.values()]
    if any(
        patient_sets[i] & patient_sets[j]
        for i in range(len(patient_sets))
        for j in range(i + 1, len(patient_sets))
    ):
        raise ValueError("Patient overlap detected between dataset splits.")

    # Save split details and counts.
    RESULTS.mkdir(parents=True, exist_ok=True)
    summary = []

    for name, split in splits.items():
        split[["patient_id", "strat_fold", "filename_hr", "labels"]].to_csv(
            RESULTS / f"{name}_recordings.csv",
            index_label="ecg_id",
        )

        summary.append({
            "split": name,
            "recordings": len(split),
            "patients": split["patient_id"].nunique(),
            **{
                label: sum(label in labels for labels in split["labels"])
                for label in CLASSES
            },
        })

    pd.DataFrame(summary).to_csv(
        RESULTS / "dataset_summary.csv",
        index=False,
    )

    # Shuffle training data only.
    return tuple(
        DataLoader(
            ECGDataset(split),
            batch_size=batch_size,
            shuffle=name == "train",
        )
        for name, split in splits.items()
    )

# test if working 
if __name__ == "__main__":
    train_loader, val_loader, test_loader = create_dataloaders(batch_size=4)

    x, y, ids = next(iter(train_loader))

    assert x.shape[1:] == (12, 5000)
    assert y.shape[1:] == (5,)

    print("Data loader is working correctly.")
    print("ECG batch shape:", x.shape)
    print("Label batch shape:", y.shape)
    print("First ECG ID:", ids[0].item())
