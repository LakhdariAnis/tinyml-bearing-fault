import numpy as np
from load_data import load_all_files

PREVIEW_LENGTH = 2000  # how many raw readings to show per file, not all 480k

def save_preview_csv(signals, filepath='data/preview.csv', n=PREVIEW_LENGTH):
    labels = list(signals.keys())
    columns = [signals[label][:n] for label in labels]
    data = np.column_stack(columns)
    header = ','.join(labels)
    np.savetxt(filepath, data, delimiter=',', header=header, comments='')
    print(f"Saved preview to {filepath}")

if __name__ == '__main__':
    signals = load_all_files()
    save_preview_csv(signals)
