import numpy as np
from load_data import load_all_files

def make_windows(signal, window_size=2048, step=1024):
    windows = []
    for start in range(0, len(signal) - window_size + 1, step):
        window = signal[start:start + window_size]
        windows.append(window)
    return np.array(windows)

if __name__ == '__main__':
    signals = load_all_files()

    all_windows = {}
    for label, signal in signals.items():
        windows = make_windows(signal, window_size=2048, step=1024)
        all_windows[label] = windows
        print(f"{label}: {windows.shape}")

    np.savez('data/windows.npz', **all_windows)
    print("\nSaved to data/windows.npz")
