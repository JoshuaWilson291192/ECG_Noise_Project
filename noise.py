# Add noise to 12-lead ECGs (12 leads x 5000 samples at 500 Hz) at a chosen signal-to-noise ratio (SNR).
# Noise types: baseline, muscle and electrode (from the NSTDB recordings), plus powerline and gaussian.
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

SAMPLE_RATE = 500
EXPECTED_SHAPE = (12, 5000)
NSTDB_RECORDS = {"baseline": "bw", "muscle": "ma", "electrode": "em"}
NOISE_TYPES = (*NSTDB_RECORDS, "powerline", "gaussian")
SNR_LEVELS = (18, 12, 6, 0, -6)


def load_nstdb(folder="data/nstdb"):
    # Load the bw, ma and em noise recordings once (needs their .hea and .dat files) and convert 360 Hz to 500 Hz
    import wfdb

    bank = {}
    for kind, name in NSTDB_RECORDS.items():
        signal, fields = wfdb.rdsamp(str(Path(folder) / name))
        if fields["fs"] != 360:
            raise ValueError(f"Expected NSTDB record {name} at 360 Hz; got {fields['fs']}.")
        bank[kind] = resample_poly(signal, up=25, down=18, axis=0).astype(np.float32)  # 360 * 25 / 18 = 500
    return bank


def _noise_template(shape, noise_type, rng, noise_bank):
    # Make raw noise for each lead; add_noise() scales it to the right strength afterwards
    n_leads, n_samples = shape
    if noise_type == "gaussian":
        return rng.standard_normal(shape).astype(np.float32)
    if noise_type == "powerline":  # 50 Hz sine wave with a random start point (phase) per lead
        time = np.arange(n_samples) / SAMPLE_RATE
        phases = rng.uniform(0, 2 * np.pi, size=(n_leads, 1))
        return np.sin(2 * np.pi * 50 * time + phases).astype(np.float32)

    if noise_type not in (noise_bank or {}):
        raise ValueError(f"Noise type '{noise_type}' needs NSTDB recordings: pass noise_bank=load_nstdb().")
    recording = np.asarray(noise_bank[noise_type], dtype=np.float32)

    # NSTDB has 2 noise channels but we need 12 leads: per lead, pick a random channel and a random 10-second slice
    template = np.empty(shape, dtype=np.float32)
    for lead in range(n_leads):
        channel = rng.integers(recording.shape[1])
        start = rng.integers(recording.shape[0] - n_samples + 1)
        segment = recording[start : start + n_samples, channel]
        template[lead] = segment - segment.mean()  # remove the constant offset, keep slow baseline drift
    return template


def add_noise(ecg, noise_type, snr_db, seed=42, noise_bank=None):
    # Return a noisy COPY of the ECG at the chosen SNR. The same seed and inputs always give the same noise
    # so use a different seed per ECG/repeat, and the same noisy ECG for every model.
    clean = np.asarray(ecg, dtype=np.float32)
    if clean.shape != EXPECTED_SHAPE:
        raise ValueError(f"Expected ECG shape {EXPECTED_SHAPE}; got {clean.shape}")
    if noise_type not in NOISE_TYPES:
        raise ValueError(f"Unknown noise type '{noise_type}'. Choose from {NOISE_TYPES}")

    template = _noise_template(clean.shape, noise_type, np.random.default_rng(seed), noise_bank)
    signal_power = np.mean(np.square(clean.astype(np.float64)))
    template_power = np.mean(np.square(template.astype(np.float64)))
    # SNR(dB) = 10 log10(P_signal / P_noise), so the noise power we want is P_signal / 10^(SNR/10)
    scale = np.sqrt(signal_power / 10 ** (snr_db / 10) / template_power)
    return clean + (template * scale).astype(np.float32)


def measure_snr_db(clean, noisy):
    # Measure the SNR using the noise that was actually added to the ECG
    clean = np.asarray(clean, dtype=np.float64)
    noisy = np.asarray(noisy, dtype=np.float64)
    if clean.shape != noisy.shape:
        raise ValueError("clean and noisy must have matching shapes")
    noise_power = np.mean((noisy - clean) ** 2)
    return float("inf") if noise_power == 0 else float(10 * np.log10(np.mean(clean**2) / noise_power))
