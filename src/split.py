import numpy as np

data = np.load("data/features.npz")

def chronological_split(feats, train_ratio=0.7, val_ratio=0.1):
    n = len(feats)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    train = feats[:train_end]
    val = feats[train_end:val_end]
    test = feats[val_end:]
    return train, val, test

label_names = data.files
label_to_int = {name: i for i, name in enumerate(label_names)}

X_train_list, y_train_list = [], []
X_val_list, y_val_list = [], []
X_test_list, y_test_list = [], []

for label in label_names:
    feats = data[label]
    train, val, test = chronological_split(feats)

    X_train_list.append(train)
    y_train_list.append(np.full(len(train), label_to_int[label]))

    X_val_list.append(val)
    y_val_list.append(np.full(len(val), label_to_int[label]))

    X_test_list.append(test)
    y_test_list.append(np.full(len(test), label_to_int[label]))

X_train = np.concatenate(X_train_list, axis=0)
y_train = np.concatenate(y_train_list, axis=0)
X_val = np.concatenate(X_val_list, axis=0)
y_val = np.concatenate(y_val_list, axis=0)
X_test = np.concatenate(X_test_list, axis=0)
y_test = np.concatenate(y_test_list, axis=0)

print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_val:   {X_val.shape}, y_val:   {y_val.shape}")
print(f"X_test:  {X_test.shape}, y_test:  {y_test.shape}")

np.savez("data/split.npz",
         X_train=X_train, y_train=y_train,
         X_val=X_val, y_val=y_val,
         X_test=X_test, y_test=y_test,
         label_names=np.array(label_names))
