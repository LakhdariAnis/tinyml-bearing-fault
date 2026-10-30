from scipy.io import loadmat
import numpy as np
import os

FILES = {
    'Time_Normal_1_098.mat': 'Normal',
    'B007_1_123.mat': 'Ball_007',
    'B014_1_190.mat': 'Ball_014',
    'B021_1_227.mat': 'Ball_021',
    'IR007_1_110.mat': 'IR_007',
    'IR014_1_175.mat': 'IR_014',
    'IR021_1_214.mat': 'IR_021',
    'OR007_6_1_136.mat': 'OR_007',
    'OR014_6_1_202.mat': 'OR_014',
    'OR021_6_1_239.mat': 'OR_021',
}

DATA_DIR = 'data/raw'

def load_one_file(filepath):
    """Load a single .mat file, return just the DE (Drive End) signal."""
    mat = loadmat(filepath)

    de_key = [k for k in mat.keys() if k.endswith('DE_time')]
    if not de_key:
        raise ValueError(f"No DE_time key found in {filepath}")

    return mat[de_key[0]].flatten()

def load_all_files():
    """Load every raw file. Returns dict: label -> signal array."""
    signals = {}
    for filename, label in FILES.items():
        filepath = os.path.join(DATA_DIR, filename)
        signal = load_one_file(filepath)
        signals[label] = signal
        print(f"Loaded {label}: {signal.shape[0]} readings")
    return signals

if __name__ == '__main__':
    signals = load_all_files()
    print(f"\nDone. Loaded {len(signals)} files.")
