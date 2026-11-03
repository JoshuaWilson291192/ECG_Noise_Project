from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from cnn import ECG_CNN
from data_loader import create_dataloaders

#setup
TEST = "--test" in sys.argv
torch.manual_seed(42)
np.random.seed(42)

folder = Path(__file__).resolve().parent
(folder / "models").mkdir(exist_ok=True)
(folder / "results").mkdir(exist_ok=True)

train_loader, val_loader, _ = create_dataloaders(batch_size=4 if TEST else 16)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ECG_CNN().to(device)

loss_fn = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("Device:", device, "Test mode:", TEST)

#train and validation
def run_epoch(loader, training):
    model.train() if training else model.eval()

    total_loss = 0
    total_ecgs = 0
    all_labels, all_predictions = [], []

    with torch.set_grad_enabled(training):
        for i, (signals)

#run