# Check the ECG noise generator and save example plots
# Use one ECG from validation fold 9 without running models or reading fold 10
# Run with python test_noise.py
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from data_loader import create_dataloaders
from noise import NOISE_TYPES, SNR_LEVELS, add_noise, load_nstdb, measure_snr_db

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "results" / "noise_checks"


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    print("Loading NSTDB baseline, muscle and electrode noise...")
    # Load the noise recordings once and reuse them for every check
    bank = load_nstdb(ROOT / "data" / "nstdb")

    # Read one validation ECG without reading the training or test batches
    _, val_loader, _ = create_dataloaders(batch_size=1)
    signals, _, ecg_ids = next(iter(val_loader))
    clean = signals[0].numpy().copy()
    original = clean.copy()
    print(f"Checking validation ECG ID {int(ecg_ids[0])}, shape {clean.shape}")

    for kind in NOISE_TYPES:
        for snr in SNR_LEVELS:
            noisy = add_noise(clean, kind, snr_db=snr, seed=42, noise_bank=bank)
            # The same seed should repeat the noise and a new seed should change it
            repeated = add_noise(clean, kind, snr_db=snr, seed=42, noise_bank=bank)
            changed = add_noise(clean, kind, snr_db=snr, seed=43, noise_bank=bank)

            # Check the shape, values and repeatability of the noisy ECG
            assert noisy.shape == clean.shape, f"Wrong shape for {kind}"
            assert np.isfinite(noisy).all(), f"Non-finite values for {kind}"
            assert np.array_equal(noisy, repeated), f"Not repeatable for {kind}"
            assert not np.array_equal(noisy, changed), f"Seed has no effect for {kind}"
            # Check the noise strength is close to the requested value
            measured = measure_snr_db(clean, noisy)
            assert abs(measured - snr) < 0.05, (kind, snr, measured)
            # Adding noise should leave the original ECG unchanged
            assert np.array_equal(clean, original), "Original ECG was modified"
            print(f"{kind:10s} requested {snr:3d} dB -> measured {measured:7.3f} dB  PASSED")

        # Plot lead II with three noise strengths to compare against the original
        t = np.arange(clean.shape[1]) / 500.0
        fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
        axes[0].plot(t, clean[1], linewidth=0.8)
        axes[0].set_title("Original ECG — lead II")
        for ax, snr in zip(axes[1:], (18, 6, -6)):
            noisy = add_noise(clean, kind, snr_db=snr, seed=42, noise_bank=bank)
            ax.plot(t, noisy[1], linewidth=0.8)
            ax.set_title(f"{kind.capitalize()} noise: {snr:+d} dB")
        for ax in axes:
            ax.set_ylabel("Amplitude (mV)")
            ax.grid(alpha=0.2)
        axes[-1].set_xlabel("Time (seconds)")
        fig.suptitle(f"ECG noise example — {kind}", fontsize=14)
        fig.tight_layout()
        fig.savefig(OUTPUT / f"{kind}_example.png", dpi=150)
        plt.close(fig)

    print("\nAll 25 noise/SNR checks PASSED. Test fold 10 was NOT used.")
    print("Figures saved at:", OUTPUT)


if __name__ == "__main__":
    main()
