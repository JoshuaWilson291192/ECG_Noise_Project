import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    multilabel_confusion_matrix,
    roc_auc_score,
    roc_curve,
)

from cnn import ECG_CNN
from resnet import ECG_ResNet
from data_loader import create_dataloaders

# Keep this order the same as the five outputs from each model
CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "results" / "evaluation_val"



def load_model(model_class, checkpoint, device):
    # Build the model, load its trained weights, and switch to evaluation mode

    model = model_class()

    # Read the saved training weights onto the CPU before choosing where to run
    weights = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=False
    )

    model.load_state_dict(weights)

    # Evaluation mode turns off dropout and uses the saved batch norm statistics
    return model.to(device).eval()



def predict(model, loader, device):
 
    ids, labels, probs = [], [], []
    # We only need predictions, so skip tracking gradients used during training
    with torch.no_grad():
        for signals, y, ecg_ids in loader:
            # Each diagnosis gets its own probability: an ECG can have several labels
            probs.append(torch.sigmoid(model(signals.to(device))).cpu().numpy())
            labels.append(y.numpy())
            ids += ecg_ids.tolist()
    # Join the batches into one set of results, keeping their original order
    return ids, np.concatenate(labels).astype(int), np.concatenate(probs)


def evaluate(labels, probs):
    # Choose a separate yes/no cutoff for each diagnosis using the validation data
    thresholds = []
    for i in range(len(CLASSES)):
        fpr, tpr, candidates = roc_curve(labels[:, i], probs[:, i])
        # Pick the best gap between correct detections and false alarms
        # Ignore the infinite cutoff returned by roc_curve
        j = np.where(np.isfinite(candidates), tpr - fpr, -1)
        thresholds.append(candidates[j.argmax()])
    thresholds = np.array(thresholds)

    # A probability at or above the cutoff counts as a positive prediction
    predicted = (probs >= thresholds).astype(int)
    matrices = multilabel_confusion_matrix(labels, predicted)

    rows = []
    for i, c in enumerate(CLASSES):
        # Count correct negatives, false alarms, missed cases and correct positives
        tn, fp, fn, tp = matrices[i].ravel()
        # Sensitivity measures cases found; specificity measures negatives ruled out
        rows.append({
            "class": c, "threshold": thresholds[i],
            "auroc": roc_auc_score(labels[:, i], probs[:, i]),
            "sensitivity": tp / (tp + fn), "specificity": tn / (tn + fp),
            "TP": tp, "FP": fp, "TN": tn, "FN": fn,
            "positive_recordings": int(labels[:, i].sum()),
        })
    return thresholds, predicted, matrices, pd.DataFrame(rows)


def save_confusion_matrices(matrices, name):
    # Save one labelled 2x2 image per diagnosis
    for c, matrix in zip(CLASSES, matrices):
        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay(matrix, display_labels=["No", "Yes"]).plot(ax=ax, values_format="d", colorbar=False)
        ax.set(title=f"{name.upper()} — {c} (validation fold 9)", xlabel="Predicted label", ylabel="Actual label")
        fig.tight_layout()
        fig.savefig(OUTPUT / f"{name}_{c}_confusion.png", dpi=160)
        # Release the figure so plots do not build up in memory.
        plt.close(fig)


def main():
    # Read optional settings from the command line
    parser = argparse.ArgumentParser(description="Validate ECG classifiers on PTB-XL fold 9")
    parser.add_argument("--dml", action="store_true", help="Use AMD GPU with DirectML")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    
    device = torch.device("cpu")
    if args.dml:
        import torch_directml

        device = torch_directml.device()

    OUTPUT.mkdir(parents=True, exist_ok=True)
    # Score fold 9 only. Leave fold 10 untouched for the final test
    _, val_loader, _ = create_dataloaders(batch_size=args.batch_size)

    saved = {
        "source": "PTB-XL validation fold 9 only",
        "method": "Maximise Youden's J = sensitivity + specificity - 1 per class",
        "classes": CLASSES,
    }
    summaries, reference = [], None

    for name, model_class in (("cnn", ECG_CNN), ("resnet", ECG_ResNet)):
        print(f"\nEvaluating {name.upper()} on fold 9 with {device}...", flush=True)
        model = load_model(model_class, ROOT / "models" / f"{name}_best.pth", device)
        ids, labels, probs = predict(model, val_loader, device)

        # Check both models see exactly the same ECGs and labels in the same order
        if reference is None:
            reference = (ids, labels)
        elif ids != reference[0] or not np.array_equal(labels, reference[1]):
            raise RuntimeError("CNN and ResNet validation ECGs are not aligned!")

        thresholds, predicted, matrices, metrics = evaluate(labels, probs)
        metrics.to_csv(OUTPUT / f"{name}_validation_metrics.csv", index=False)
        save_confusion_matrices(matrices, name)

        # Save one row per ECG with its true labels, probabilities and decisions
        pd.DataFrame({"ecg_id": ids, **{
            f"{prefix}_{c}": values[:, i]
            for i, c in enumerate(CLASSES)
            for prefix, values in (("true", labels), ("prob", probs), ("pred", predicted))
        }}).to_csv(OUTPUT / f"{name}_validation_predictions.csv", index=False)

        saved[name] = dict(zip(CLASSES, thresholds.tolist()))
        # Give each diagnosis equal weight when averaging the scores
        macro = metrics[["auroc", "sensitivity", "specificity"]].mean().add_prefix("macro_")
        summaries.append({"model": name, "n_recordings": len(labels), **macro.to_dict()})

        print(metrics.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        print(f"\n{name.upper()} validation macro-AUROC: {macro['macro_auroc']:.4f}")

    # Keep the chosen cutoffs for later testing and save a model comparison table
    with open(OUTPUT / "thresholds.json", "w", encoding="utf-8") as f:
        json.dump(saved, f, indent=2)
    pd.DataFrame(summaries).to_csv(OUTPUT / "validation_summary.csv", index=False)
    print("\nVALIDATION COMPLETE — fold 10 has NOT been evaluated.")
    print("Results saved in:", OUTPUT)
    print("Selected decision thresholds saved in: thresholds.json")


if __name__ == "__main__":
    main()
