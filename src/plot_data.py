"""
Compare vibration signals across load conditions (1HP vs 3HP),
normal vs defective, in both time and frequency domain.
"""

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import os

# ----------------------------------------------------------------
# Config
# ----------------------------------------------------------------
FS = 12000  # sampling rate for CWRU Drive-End 12kHz files — verify this matches your files
N_SAMPLES = 2000  # window length for time-domain plots
FFT_XLIM = 500  # Hz, upper limit for FFT x-axis (bearing faults live below this)

paths = {
    "normal_1hp": "data/raw/Time_Normal_1_098.mat",
    "defect_1hp": "data/raw/B007_1_123.mat",
    "normal_3hp": "data/raw_3/Normal_3_100.mat",
    "defect_3hp": "data/raw_3/B007_3_125.mat",
}

os.makedirs("plots", exist_ok=True)


# ----------------------------------------------------------------
# Loading
# ----------------------------------------------------------------
def load_de_signal(mat_path):
    """Load the Drive-End accelerometer signal from a CWRU .mat file."""
    data = sio.loadmat(mat_path)
    de_key = [k for k in data.keys() if k.endswith('_DE_time')][0]
    signal = data[de_key].flatten()
    return signal


signals = {name: load_de_signal(p) for name, p in paths.items()}


# ----------------------------------------------------------------
# Sanity check — catch flat/empty signals before we waste time plotting
# ----------------------------------------------------------------
print("Signal stats:")
for name, sig in signals.items():
    print(f"  {name}: len={len(sig)}, mean={sig.mean():.4f}, std={sig.std():.4f}, "
          f"min={sig.min():.4f}, max={sig.max():.4f}")


# ----------------------------------------------------------------
# FFT helper
# ----------------------------------------------------------------
def compute_fft(signal, fs=FS):
    signal = signal - np.mean(signal)   # remove DC offset so it doesn't dominate the scale
    n = len(signal)
    freqs = np.fft.rfftfreq(n, d=1 / fs)
    mag = np.abs(np.fft.rfft(signal)) / n
    return freqs, mag


layout = [
    ("normal_1hp", "Normal - 1HP", "tab:blue"),
    ("defect_1hp", "Defect (B007) - 1HP", "tab:red"),
    ("normal_3hp", "Normal - 3HP", "tab:blue"),
    ("defect_3hp", "Defect (B007) - 3HP", "tab:red"),
]


# ----------------------------------------------------------------
# Plot 1: time domain, 2x2 grid
# ----------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 6), sharex=True, sharey=True)
axes_flat = axes.flat

for (key, title, color), ax in zip(layout, axes_flat):
    ax.plot(signals[key][:N_SAMPLES], color=color)
    ax.set_title(title)
    ax.set_xlabel("Sample")
    ax.set_ylabel("Amplitude")

plt.tight_layout()
plt.savefig("plots/time_domain_1hp_vs_3hp.png", dpi=150)
plt.show()


# ----------------------------------------------------------------
# Plot 2: frequency domain, 2x2 grid
# ----------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 6))
axes_flat = axes.flat

for (key, title, color), ax in zip(layout, axes_flat):
    freqs, mag = compute_fft(signals[key])
    ax.plot(freqs[1:], mag[1:], color=color)   # skip index 0 (DC bin)
    ax.set_xlim(0, FFT_XLIM)
    ax.set_title(f"{title} (FFT)")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")

plt.tight_layout()
plt.savefig("plots/fft_1hp_vs_3hp.png", dpi=150)
plt.show()

print("\nSaved plots/time_domain_1hp_vs_3hp.png and plots/fft_1hp_vs_3hp.png")
