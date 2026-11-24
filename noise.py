# Add noise to 12 lead ECGs with 5000 samples at 500 Hz
# Support baseline, muscle, electrode, powerline and Gaussian noise
# Measure signal and noise power across all leads
# Keep the original ECG unchanged
# Load the NSTDB recordings once and reuse them
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

SAMPLE_RATE = 500
EXPECTED_SHAPE = (12, 5000)
NSTDB_RECORDS = {
    "baseline": "bw",
    "muscle": "ma",
    "electrode": "em",
}
NOISE_TYPES = (*NSTDB_RECORDS, "powerline", "gaussian")
SNR_LEVELS = (18, 12, 6, 0, -6)


def load_nstdb(folder="data/nstdb"):
    # Load the three noise recordings and convert 360 Hz to 500 Hz
    # Needs the hea and dat files for bw, ma and em
    # Return two noise channels per recording
    import wfdb

    folder = Path(folder)
    bank = {}
    for kind, record_name in NSTDB_RECORDS.items():
        record_path = folder / record_name
        try:
            physical_signals, fields = wfdb.rdsamp(str(record_path))
        except (FileNotFoundError, OSError) as exc:
            raise FileNotFoundError(
                f"Cannot read {record_name}.hea/.dat from {folder}. "
                "Download the NSTDB bw, ma and em records first."
            ) from exc

        source_fs = float(fields["fs"])
        if source_fs != 360.0:
            raise ValueError(
                f"Expected NSTDB record {record_name} at 360 Hz; got {source_fs}."
            )
        # 360 * 25 / 18 = 500 samples per second
        resampled = resample_poly(physical_signals, up=25, down=18, axis=0)
        resampled = np.asarray(resampled, dtype=np.float32)
        if resampled.ndim != 2 or resampled.shape[0] < EXPECTED_SHAPE[1]:
            raise ValueError(f"Unexpected waveform shape in {record_name}: {resampled.shape}")
        if not np.isfinite(resampled).all():
            raise ValueError(f"NSTDB record {record_name} contains non-finite samples")
        bank[kind] = resampled
    return bank


def _noise_template(shape, noise_type, rng, noise_bank):
    # Create noise for each ECG lead before adjusting its strength
    n_leads, n_samples = shape
    if noise_type == "gaussian":
        return rng.standard_normal(shape).astype(np.float32)

    if noise_type == "powerline":
        time = np.arange(n_samples, dtype=np.float64) / SAMPLE_RATE
        phases = rng.uniform(0, 2 * np.pi, size=(n_leads, 1))
        return np.sin(2 * np.pi * 50 * time[None, :] + phases).astype(np.float32)

    if noise_bank is None or noise_type not in noise_bank:
        raise ValueError(
            f"Noise type '{noise_type}' needs NSTDB recordings. "
            "Call load_nstdb('data/nstdb') and pass noise_bank=bank."
        )

    recording = np.asarray(noise_bank[noise_type], dtype=np.float32)
    if recording.ndim != 2 or recording.shape[0] < n_samples:
        raise ValueError(f"Invalid NSTDB data for {noise_type}: {recording.shape}")

    template = np.empty(shape, dtype=np.float32)
    # NSTDB has two noise channels, whereas PTB-XL has twelve ECG leads
    # Choose a channel and a random contiguous 10-second segment per lead
    for lead in range(n_leads):
        channel = int(rng.integers(recording.shape[1]))
        start = int(rng.integers(recording.shape[0] - n_samples + 1))
        segment = recording[start : start + n_samples, channel].copy()
        # Remove the constant offset but keep the slow baseline changes
        template[lead] = segment - segment.mean()
    return template


def add_noise(ecg, noise_type, snr_db, seed=42, noise_bank=None):
    # Return a noisy copy of the ECG at the chosen signal to noise ratio
    # The same seed and inputs give the same noise
    # Use a different seed per ECG and repeat, and the same noisy ECG for all models
    clean = np.asarray(ecg, dtype=np.float32)
    if clean.shape != EXPECTED_SHAPE:
        raise ValueError(f"Expected ECG shape {EXPECTED_SHAPE}; got {clean.shape}")
    if not np.isfinite(clean).all():
        raise ValueError("ECG contains NaN or infinity")
    if noise_type not in NOISE_TYPES:
        raise ValueError(f"Unknown noise type '{noise_type}'. Choose from {NOISE_TYPES}")
    if not np.isfinite(snr_db):
        raise ValueError("snr_db must be a finite number")

    rng = np.random.default_rng(seed)
    template = _noise_template(clean.shape, noise_type, rng, noise_bank)
    clean_power = float(np.mean(np.square(clean.astype(np.float64))))
    template_power = float(np.mean(np.square(template.astype(np.float64))))
    if clean_power <= 0 or template_power <= 0:
        raise ValueError("ECG or noise template has zero power; cannot specify SNR")

    # SNR(dB) = 10 log10(P_signal / P_added_noise)
    desired_noise_power = clean_power / (10.0 ** (float(snr_db) / 10.0))
    scale = np.sqrt(desired_noise_power / template_power)
    return clean + (template * scale).astype(np.float32)


def measure_snr_db(clean, noisy):
    # Measure the signal to noise ratio using the noise added to the ECG
    clean = np.asarray(clean, dtype=np.float64)
    noisy = np.asarray(noisy, dtype=np.float64)
    if clean.shape != noisy.shape:
        raise ValueError("clean and noisy must have matching shapes")
    added = noisy - clean
    signal_power = np.mean(clean**2)
    noise_power = np.mean(added**2)
    if noise_power == 0:
        return float("inf")
    return float(10 * np.log10(signal_power / noise_power))
