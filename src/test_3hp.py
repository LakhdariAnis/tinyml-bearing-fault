import re
import numpy as np
from scipy.io import loadmat
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix

from windowing import make_windows
from features import extract_time_features, extract_freq_features

WINDOW_SIZE = 2048
STEP = 1024
RAW3_DIR = "data/raw_3"

# Real filenames from your data/raw_3 directory, mapped to the SAME
# label strings used in load-1 (must match label_names in split.npz exactly).
FILES_LOAD3 = {
    "Normal_3_100.mat":     "Normal",
    "B007_3_125.mat":       "Ball_007",
    "B014_3_192.mat":       "Ball_014",
    "B021_3_229.mat":       "Ball_021",
    "IR007_3_112.mat":      "IR_007",
    "IR014_3_177.mat":      "IR_014",
    "IR021_3_217.mat":      "IR_021",   # double-size file, handled below
    "OR007_6_3_138.mat":    "OR_007",
    "OR014_6_3_204.mat":    "OR_014",
    "OR021_6_3_241.mat":    "OR_021",
}


def get_de_signal(mat_dict, filename):
    """
    Pick the DE_time array whose numeric ID matches the filename's ID.
    Needed because IR021_3_217.mat contains TWO bundled recordings
    (X215_* and X217_*) — only 217 is the one labeled IR021 by the
    filename; X215_* belongs to a different, unlabeled recording and
    must be ignored, not concatenated in.
    """
    file_id = re.search(r'_(\d+)\.mat$', filename).group(1)
    expected_key = f"X{file_id}_DE_time"
    if expected_key not in mat_dict:
        raise ValueError(
            f"{filename}: expected key {expected_key} not found. "
            f"Available DE keys: {[k for k in mat_dict if 'DE_time' in k]}"
        )
    return mat_dict[expected_key].flatten()


def build_windows_by_label():
    """
    Load each load-3 raw file, extract the correct DE signal, window it.
    Returns dict: label -> array of shape (n_windows, window_size),
    same structure as data/windows.npz, so extract_time_features /
    extract_freq_features work unmodified.
    """
    windows_by_label = {}
    for filename, label in FILES_LOAD3.items():
        mat = loadmat(f"{RAW3_DIR}/{filename}")
        signal = get_de_signal(mat, filename)
        windows = make_windows(signal, window_size=WINDOW_SIZE, step=STEP)
        windows_by_label[label] = windows
        print(f"{label}: {windows.shape}")
    return windows_by_label


def build_feature_matrix(windows_by_label, label_names):
    """
    Run the SAME feature extraction as features.py, batched per label,
    then stack into one X matrix and one y (integer) label array,
    with y indices matching label_names order from split.npz — this is
    what lets predictions line up with the model's trained class indices.
    """
    X_parts = []
    y_parts = []
    for label, windows in windows_by_label.items():
        feats = np.concatenate([
            extract_time_features(windows),
            extract_freq_features(windows),
        ], axis=1)
        X_parts.append(feats)
        y_parts.append(np.full(len(feats), label_names.index(label)))

    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    return X, y


def main():
    print("Loading + windowing load-3 raw files...")
    windows_by_label = build_windows_by_label()

    # Save separately from load-1's features.npz — never overwrite it.
    np.savez("data/features_load3.npz", **windows_by_label)
    print("Saved windows to data/windows_load3_raw (in features_load3.npz for now)\n")

    split_data = np.load("data/split.npz", allow_pickle=True)
    label_names = list(split_data["label_names"])

    print("Extracting features...")
    X_raw, y_true = build_feature_matrix(windows_by_label, label_names)
    print(f"X shape: {X_raw.shape}\n")

    # Normalize using load-1's SAVED train stats — never recompute from load-3.
    stats = np.load("models/normalization_stats.npz")
    X_norm = (X_raw - stats["feature_mean"]) / stats["feature_std"]

    model = keras.models.load_model("models/baseline.keras")
    y_pred = np.argmax(model.predict(X_norm), axis=1)

    print("\n=== Cross-condition (load 3) results ===")
    print(classification_report(y_true, y_pred, target_names=label_names))
    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))


if __name__ == "__main__":
    main()
