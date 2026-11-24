# Compare both models on clean and noisy ECGs
# Use validation fold 9 first and fold 10 for the final test
# Keep the saved cutoffs and give both models the same noise
# Reuse saved ECGs and save progress without training again

import argparse
import json
import os
from collections import Counter
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cnn import ECG_CNN
from data_loader import create_dataloaders
from sklearn.metrics import multilabel_confusion_matrix, roc_auc_score
from noise import NOISE_TYPES, SNR_LEVELS, add_noise, load_nstdb, measure_snr_db
from resnet import ECG_ResNet

CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data" / "cache"
THRESHOLDS = ROOT / "results" / "evaluation_val" / "thresholds.json"
MODEL_CLASSES = {"cnn": ECG_CNN, "resnet": ECG_ResNet}
REPEATS = (42, 43, 44)


def load_model(model_class, checkpoint, device):
    # Load the trained model
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint}")
    model = model_class()
    # Only load model files you trust
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False))
    return model.to(device).eval()


def calculate_metrics(labels, probs, thresholds):
    # Calculate scores for each diagnosis and the overall average
    if labels.shape != probs.shape or labels.ndim != 2 or labels.shape[1] != len(CLASSES):
        raise ValueError("Labels and probabilities must both have shape (N, 5)")
    if thresholds.shape != (len(CLASSES),):
        raise ValueError("Need one fixed validation threshold for each class")
    if not np.isfinite(probs).all():
        raise ValueError("Probabilities contain NaN or infinity")
    predicted = (probs >= thresholds[None, :]).astype(int)
    matrices = multilabel_confusion_matrix(labels.astype(int), predicted)
    rows = []
    for i, label_name in enumerate(CLASSES):
        tn, fp, fn, tp = (int(x) for x in matrices[i].ravel())
        if len(np.unique(labels[:, i])) != 2:
            raise ValueError(f"AUROC requires both positive and negative examples for {label_name}")
        rows.append({
            "class": label_name,
            "threshold": float(thresholds[i]),
            "auroc": float(roc_auc_score(labels[:, i], probs[:, i])),
            "sensitivity": tp / (tp + fn) if tp + fn else float("nan"),
            "specificity": tn / (tn + fp) if tn + fp else float("nan"),
            "TP": tp, "FP": fp, "TN": tn, "FN": fn,
            "positive_recordings": int(labels[:, i].sum()),
        })
    metrics = pd.DataFrame(rows)
    summary = {"n_recordings": int(len(labels)), **{
        f"macro_{metric}": float(metrics[metric].mean())
        for metric in ("auroc", "sensitivity", "specificity")
    }}
    return metrics, matrices, summary


def load_thresholds():
    # Load the saved cutoffs from validation fold 9
    if not THRESHOLDS.is_file():
        raise FileNotFoundError(
            f"Missing {THRESHOLDS}. Run evaluate_models.py on fold 9 first."
        )
    saved = json.loads(THRESHOLDS.read_text(encoding="utf-8"))
    if saved.get("classes") != CLASSES:
        raise ValueError("Class order in thresholds.json differs from this project")
    if "fold 9" not in saved.get("source", ""):
        raise ValueError("Thresholds must be selected on validation fold 9")

    thresholds = {}
    for name in MODEL_CLASSES:
        values = np.array([saved[name][c] for c in CLASSES], dtype=float)
        if not np.isfinite(values).all() or not np.all((values >= 0) & (values <= 1)):
            raise ValueError(f"Invalid {name} thresholds: {values}")
        thresholds[name] = values
    return thresholds


def load_or_create_cache(split, batch_size):
    # Save ECGs once and reuse them on later runs
    CACHE.mkdir(parents=True, exist_ok=True)
    signals_file, labels_file, ids_file = (
        CACHE / f"{split}_{kind}.npy" for kind in ("signals", "labels", "ids")
    )
    ready_file = CACHE / f"{split}_cache_complete.txt"

    if not all(path.exists() for path in (ready_file, signals_file, labels_file, ids_file)):
        print(f"Building {split} ECG cache from WFDB files (one-time step)...", flush=True)
        _, val_loader, test_loader = create_dataloaders(batch_size=batch_size)
        loader = val_loader if split == "val" else test_loader
        n = len(loader.dataset)

        temporary_file = CACHE / f"{split}_signals_incomplete.npy"
        if temporary_file.exists():
            temporary_file.unlink()
        mapped = np.lib.format.open_memmap(
            temporary_file, mode="w+", dtype=np.float32, shape=(n, 12, 5000)
        )
        labels = np.empty((n, len(CLASSES)), dtype=np.uint8)
        ids = np.empty(n, dtype=np.int64)
        cursor = 0

        for batch_number, (signals, y, ecg_ids) in enumerate(loader, start=1):
            chunk = signals.cpu().numpy().astype(np.float32, copy=False)
            labels_chunk = y.cpu().numpy()
            ids_chunk = (ecg_ids.cpu().numpy() if torch.is_tensor(ecg_ids)
                         else np.asarray(ecg_ids, dtype=np.int64))
            count = len(chunk)
            if chunk.shape[1:] != (12, 5000) or labels_chunk.shape != (count, 5):
                raise ValueError("Unexpected shape in DataLoader")
            if not np.isfinite(chunk).all() or not np.isin(labels_chunk, [0, 1]).all():
                raise ValueError("Non-finite ECG or unexpected labels in DataLoader")
            mapped[cursor:cursor + count] = chunk
            labels[cursor:cursor + count] = labels_chunk.astype(np.uint8)
            ids[cursor:cursor + count] = ids_chunk
            cursor += count
            if batch_number % 30 == 0:
                print(f"  Cached {cursor}/{n} ECGs", flush=True)

        if cursor != n or len(np.unique(ids)) != n:
            raise ValueError("ECG cache is incomplete or has duplicate IDs")
        mapped.flush()
        del mapped
        os.replace(temporary_file, signals_file)
        np.save(labels_file, labels)
        np.save(ids_file, ids)
        ready_file.write_text("Complete: PTB-XL fold 9" if split == "val"
                              else "Complete: PTB-XL fold 10", encoding="utf-8")
        cache_size_mb = signals_file.stat().st_size / (1024 ** 2)
        print(f"Saved cache at {CACHE} ({cache_size_mb:.1f} MiB of ECG signals).", flush=True)

    signals = np.load(signals_file, mmap_mode="r", allow_pickle=False)
    labels = np.load(labels_file, allow_pickle=False)
    ids = np.load(ids_file, allow_pickle=False)
    if (signals.ndim != 3 or signals.shape[1:] != (12, 5000)
            or labels.shape != (len(signals), 5)
            or ids.shape != (len(signals),)):
        raise ValueError("Invalid cache. Delete the split's cache files and rerun.")
    return signals, labels.astype(int), ids


def condition_id(noise_type, snr, seed):
    return "original" if noise_type == "original" else f"{noise_type}_snr{snr:+d}_seed{seed}"


def predict_condition(signals, ids, models, device, batch_size,
                      noise_type, snr, seed, noise_bank):
    # Give both models the same noisy ECGs and collect their predictions
    n = len(ids)
    predictions = {name: np.empty((n, len(CLASSES)), dtype=np.float32) for name in models}
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        clean_batch = np.array(signals[start:end], dtype=np.float32, copy=True)
        if noise_type == "original":
            batch = clean_batch
        else:
            # Keep the same noise for each ECG regardless of batch size or noise strength
            batch = np.stack([
                add_noise(ecg, noise_type=noise_type, snr_db=snr, noise_bank=noise_bank,
                          seed=np.random.SeedSequence([
                              int(seed), int(ecg_id), NOISE_TYPES.index(noise_type)
                          ]))
                for ecg, ecg_id in zip(clean_batch, ids[start:end])
            ])
            if start == 0:
                measured = measure_snr_db(clean_batch[0], batch[0])
                if abs(measured - snr) > 0.05:
                    raise ValueError(f"Requested {snr} dB but measured {measured:.3f} dB")

        input_tensor = torch.from_numpy(batch).to(device)
        with torch.no_grad():
            for name, model in models.items():
                output = model(input_tensor)
                if output.shape != (end - start, len(CLASSES)):
                    raise ValueError(f"Unexpected {name} output: {tuple(output.shape)}")
                probabilities = torch.sigmoid(output).cpu().numpy()
                if not np.isfinite(probabilities).all():
                    raise ValueError(f"{name} produced NaN or infinity")
                predictions[name][start:end] = probabilities

        if (end // batch_size) % 30 == 0 or end == n:
            print(f"    {end}/{n} ECGs processed", flush=True)
    return predictions


def evaluate_condition(labels, probabilities, thresholds, split, cid, noise_type, snr, seed):
    summary_rows, class_rows = [], []
    for model_name, probs in probabilities.items():
        # Calculate scores using the fixed cutoffs
        class_metrics, matrices, summary = calculate_metrics(
            labels, probs, thresholds[model_name]
        )
        details = {
            "split": split, "condition_id": cid, "model": model_name,
            "noise_type": noise_type, "snr_db": "original" if snr is None else snr,
            "seed": "original" if seed is None else seed,
        }
        summary_rows.append({**details, **summary})
        class_rows.extend({**details, **row} for row in class_metrics.to_dict("records"))
    return summary_rows, class_rows


def save_csv_atomic(rows, path):
    # Save to a temporary file first in case the run stops
    temporary = path.with_suffix(".tmp.csv")
    pd.DataFrame(rows).to_csv(temporary, index=False)
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description="CNN/ResNet ECG noise robustness experiments")
    parser.add_argument("--dml", action="store_true", help="Run model inference on AMD DirectML")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--noise-type", choices=NOISE_TYPES, default="gaussian")
    parser.add_argument("--snr", type=int, choices=SNR_LEVELS, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--all", action="store_true", help="All five noise types x five SNRs x three repeats")
    parser.add_argument("--smoke", action="store_true", help="Check only 32 validation ECGs, no metrics")
    parser.add_argument("--confirm-test", action="store_true", help="Confirm one final test-set evaluation")
    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.split == "test" and not (args.confirm_test and args.all and not args.smoke):
        parser.error("Test fold 10 is reserved for the FINAL full run: --split test --confirm-test --all")
    if args.smoke and args.all:
        parser.error("Use --smoke by itself, not with --all")

    thresholds = load_thresholds()
    if args.dml:
        import torch_directml
        device = torch_directml.device()
    else:
        device = torch.device("cpu")
    print(f"Device: {device} | Dataset: {args.split}", flush=True)

    signals, labels, ids = load_or_create_cache(args.split, args.batch_size)
    if args.smoke:
        signals, labels, ids = signals[:32], labels[:32], ids[:32]
        print("SMOKE TEST ONLY: 32 validation ECGs; no metrics or result CSVs", flush=True)

    models = {
        name: load_model(model_class, ROOT / "models" / f"{name}_best.pth", device)
        for name, model_class in MODEL_CLASSES.items()
    }

    conditions = [("original", None, None)] + (
        list(product(NOISE_TYPES, SNR_LEVELS, REPEATS))
        if args.all else [(args.noise_type, args.snr, args.seed)]
    )

    output_folder = ROOT / "results" / f"robustness_{args.split}"
    summary_path = output_folder / "summary.csv"
    class_path = output_folder / "per_class.csv"
    if not args.smoke:
        output_folder.mkdir(parents=True, exist_ok=True)
        summary_rows, class_rows = (
            pd.read_csv(path).to_dict("records") if path.exists() else []
            for path in (summary_path, class_path)
        )
        summary_counts = Counter((row["condition_id"], row["model"]) for row in summary_rows)
        class_names = {}
        for row in class_rows:
            key = (row["condition_id"], row["model"])
            class_names.setdefault(key, []).append(row["class"])
        completed = {
            cid for cid, _ in summary_counts
            if all(
                summary_counts[(cid, name)] == 1
                and sorted(class_names.get((cid, name), [])) == sorted(CLASSES)
                for name in MODEL_CLASSES
            )
        }
        # Remove unfinished results before trying them again
        summary_rows, class_rows = (
            [row for row in rows if row["condition_id"] in completed]
            for rows in (summary_rows, class_rows)
        )
    else:
        summary_rows, class_rows, completed = [], [], set()

    need_nstdb = any(noise_type in ("baseline", "muscle", "electrode")
                     and condition_id(noise_type, snr, seed) not in completed
                     for noise_type, snr, seed in conditions)
    bank = load_nstdb(ROOT / "data" / "nstdb") if need_nstdb else None

    for index, (noise_type, snr, seed) in enumerate(conditions, start=1):
        cid = condition_id(noise_type, snr, seed)
        if cid in completed:
            print(f"[{index}/{len(conditions)}] Already completed: {cid}", flush=True)
            continue
        print(f"\n[{index}/{len(conditions)}] Evaluating: {cid}", flush=True)
        probabilities = predict_condition(
            signals, ids, models, device, args.batch_size,
            noise_type, snr, seed, bank
        )
        if args.smoke:
            for name, values in probabilities.items():
                print(f"{name.upper()} probabilities: {values.shape}; "
                      f"range {values.min():.4f} to {values.max():.4f}")
            continue

        new_summary, new_class = evaluate_condition(
            labels, probabilities, thresholds, args.split, cid, noise_type, snr, seed
        )
        summary_rows.extend(new_summary)
        class_rows.extend(new_class)
        save_csv_atomic(summary_rows, summary_path)
        save_csv_atomic(class_rows, class_path)
        for row in new_summary:
            print(f"  {row['model'].upper():6s}: macro-AUROC {row['macro_auroc']:.4f} | "
                  f"sensitivity {row['macro_sensitivity']:.4f} | "
                  f"specificity {row['macro_specificity']:.4f}")
        print("  Saved progress to CSV files.", flush=True)

    if args.smoke:
        print("\nSMOKE TEST PASSED — no test set was used, no final results saved.")
    else:
        print(f"\nFinished. Summary: {summary_path}")
        print(f"Class results: {class_path}")
        print("Thresholds were loaded from validation fold 9 (not re-selected).")
        if args.split == "val":
            print("TEST FOLD 10 WAS NOT USED.")


if __name__ == "__main__":
    main()
