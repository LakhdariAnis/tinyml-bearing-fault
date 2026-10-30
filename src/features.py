import numpy as np
from scipy.stats import kurtosis, skew

data = np.load("data/windows.npz")
SAMPLE_RATE = 48000

# Frequency bands to sum energy over, in Hz. Nyquist limit here is 24000 Hz
# (half the sample rate — the highest frequency FFT can represent).
BANDS = [(0, 2000), (2000, 5000), (5000, 10000), (10000, 24000)]

def extract_time_features(windows):
    mean = windows.mean(axis=1)
    std = windows.std(axis=1)
    rms = np.sqrt(np.mean(windows**2, axis=1))
    kurt = kurtosis(windows, axis=1)
    peak = np.max(np.abs(windows), axis=1)
    crest = peak / rms
    skewness = skew(windows, axis=1)
    return np.stack([mean, std, rms, kurt, crest, skewness], axis=1)

def extract_freq_features(windows, sample_rate=SAMPLE_RATE):
    n = windows.shape[1]
    fft_vals = np.fft.rfft(windows, axis=1)
    magnitude = np.abs(fft_vals)
    freqs = np.fft.rfftfreq(n, d=1/sample_rate)

    magnitude_no_dc = magnitude[:, 1:]
    freqs_no_dc = freqs[1:]

    dominant_idx = np.argmax(magnitude_no_dc, axis=1)
    dominant_freq = freqs_no_dc[dominant_idx]

    spectral_energy = np.sum(magnitude_no_dc**2, axis=1)

    spectral_centroid = np.sum(freqs_no_dc * magnitude_no_dc, axis=1) / np.sum(magnitude_no_dc, axis=1)

    band_features = []
    for low, high in BANDS:
        band_mask = (freqs_no_dc >= low) & (freqs_no_dc < high)
        band_energy = np.sum(magnitude_no_dc[:, band_mask]**2, axis=1)
        band_features.append(band_energy)

    return np.stack([dominant_freq, spectral_energy, spectral_centroid, *band_features], axis=1)

all_features = {}
for label in data.files:
    windows = data[label]
    feats = np.concatenate([
        extract_time_features(windows),
        extract_freq_features(windows),
    ], axis=1)
    all_features[label] = feats
    print(f"{label}: {feats.shape}")

np.savez("data/features.npz", **all_features)
print("\nSaved to data/features.npz")
