import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt


data = np.load("data/split.npz")
X_train, y_train = data["X_train"], data["y_train"]
X_val, y_val = data["X_val"], data["y_val"]
X_test, y_test = data["X_test"], data["y_test"]
label_names = data["label_names"]

# Normalize each feature using TRAIN statistics only
feature_mean = X_train.mean(axis=0)
feature_std = X_train.std(axis=0)
np.savez("models/normalization_stats.npz",
         feature_mean=feature_mean,
         feature_std=feature_std)
X_train = (X_train - feature_mean) / feature_std
X_val = (X_val - feature_mean) / feature_std
X_test = (X_test - feature_mean) / feature_std

model = keras.Sequential([
    layers.Input(shape=(13,)),
    layers.Dense(16, activation="relu"),
    layers.Dense(8, activation="relu"),
    layers.Dense(10, activation="softmax"),
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
)

plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('models/training_curve.png')
print("Saved training curve to models/training_curve.png")
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"\nFinal test accuracy: {test_acc:.4f}")

model.save("models/baseline.keras")
print("Saved model to models/baseline.keras")

# Per-class breakdown
y_pred = model.predict(X_test).argmax(axis=1)
print("\nPer-class performance:")
print(classification_report(y_test, y_pred, target_names=label_names))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, index=label_names, columns=label_names)
print("\nConfusion matrix (rows = actual, columns = predicted):")
print(cm_df)
