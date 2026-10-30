import numpy as np
import tensorflow as tf
from tensorflow import keras

# Load the trained float32 model — this is what we're shrinking.
model = keras.models.load_model("models/baseline.keras")

# Load the ORIGINAL split — X_test here is the same 20% held-out load-1
# data that gave you 98%. We reuse it, unchanged, so the INT8 accuracy
# number is directly comparable to your float32 number.
split_data = np.load("data/split.npz", allow_pickle=True)
X_train_raw = split_data["X_train"]
X_test_raw = split_data["X_test"]
y_test = split_data["y_test"]

# Load the SAME normalization stats the float32 model was evaluated with.
# Quantization calibration and test-time normalization must use these,
# not stats recomputed from anything — consistent with your existing rule.
stats = np.load("models/normalization_stats.npz")
feature_mean = stats["feature_mean"]
feature_std = stats["feature_std"]

X_train_norm = (X_train_raw - feature_mean) / feature_std
X_test_norm = (X_test_raw - feature_mean) / feature_std


def representative_dataset():
    """
    TFLite's INT8 converter needs to see realistic input examples to decide
    the numeric range (min/max) each layer's activations should be scaled to.
    This is called 'calibration'. It does NOT retrain anything — it just
    watches activations flow through the float model on real data and
    records their ranges, so it knows how to map float -> int8 later.

    We use X_train (normalized), not X_test — calibrating on test data
    would be a form of leakage: letting the compression process "see"
    the data it'll later be graded on.
    """
    for i in range(len(X_train_norm)):
        # TFLite expects each sample as a batch of 1, shape (1, 13), float32.
        sample = X_train_norm[i:i+1].astype(np.float32)
        yield [sample]


# Set up the converter, starting from your saved Keras model.
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# DEFAULT optimization enables quantization; combined with the settings
# below, it produces FULL integer quantization (weights AND activations
# AND input/output all INT8) — required for TFLite Micro, which has no
# floating-point-heavy runtime support on most MCUs.
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset

# Restrict the converter to INT8-only ops. Without this, it may fall back
# to float32 for ops it can't quantize, silently defeating the point.
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

# Force the MODEL'S input and output tensors to be int8 too (not just
# internal layers). This matters because your MCU code will feed raw
# int8 bytes in and read int8 bytes out — no float32 conversion on-device.
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()

with open("models/baseline_int8.tflite", "wb") as f:
    f.write(tflite_model)

print(f"Saved INT8 model: {len(tflite_model)} bytes "
      f"(float32 .keras was larger — check with `ls -la models/`)")
