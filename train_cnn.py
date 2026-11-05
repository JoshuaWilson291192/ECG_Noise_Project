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


USE_DML = "--dml" in sys.argv

if USE_DML:
    import torch_directml
    device = torch_directml.device()
else:
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

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
        for i, (signals, labels, _) in enumerate(loader):
            if TEST and i == 2:  # Limit batches in test mode
                break

            signals = signals.to(device)
            labels = labels.to(device)

            if training:
                optimizer.zero_grad()

            logits = model(signals)
            loss = loss_fn(logits, labels)

            if training:
                loss.backward()
                optimizer.step()
            else:
                all_labels.append(labels.cpu().numpy())
                all_predictions.append(torch.sigmoid(logits).cpu().numpy())

            total_loss += loss.item() * len(signals)
            total_ecgs += len(signals)

    average_loss = total_loss / total_ecgs
    auc = float('nan')

    if not training and not TEST:
        labels = np.concatenate(all_labels)
        probabilities = np.concatenate(all_predictions)

        auc = roc_auc_score(labels, probabilities, average="macro")
        

    return average_loss, auc

#run
best_loss = float('inf')
no_improvement = 0
history = []

max_epochs = 1 if TEST else 20

model_file = folder / "models" / ("cnn_test.pth" if TEST else "cnn_best.pth")

history_file = folder / "results" / ("history_test.csv" if TEST else "history_final.csv")

for epoch in range(1, max_epochs + 1):
    print(f"Starting epoch {epoch}/{max_epochs}...")

    train_loss, _ = run_epoch(train_loader, training=True)
    val_loss, val_auc = run_epoch(val_loader, training=False)

    print (
        f"Epoch {epoch}: Train Loss = {train_loss:.4f},"
        f"Val Loss ={val_loss:.4f},"
        f"Val macro-AUC={val_auc:.4f}"
    )

    history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_auc": val_auc})

    pd.DataFrame(history).to_csv(history_file, index=False)

    # save model with lowest validation loss
    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model.state_dict(), model_file)
        no_improvement = 0
        print("Saved best model:", model_file.name)

    else:
        no_improvement += 1

        if no_improvement >= 4:
            print("No improvement for 4 epochs. Stopping training.")
            break

print("Training complete. Training History:", history_file)

if TEST:
    print("TEST MODE - NOT A TRAINED MODEL")