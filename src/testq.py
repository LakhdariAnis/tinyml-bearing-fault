import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

# Load the SAME test split used for the float32 result — this keeps
# the comparison apples-to-apples: same 20% held-out load-1 windows.
split_data = np.load("data/split.npz", allow_pickle=True)
X_test_raw = split_data["X_test"]
y_test = split_data["y_test"]
label_names = list(split_data["label_names"])

stats = np.load("models/normalization_stats.npz")
feature_mean = stats["feature_mean"]
feature_std = stats["feature_std"]

X_test_norm = (X_test_raw - feature_mean) / feature_std

# --- Load the TFLite model into an "interpreter" ---
# Unlike Keras's model.predict(), TFLite has no simple .predict().
# Memory buffers ("tensors") are allocated once, then data goes into the
# input tensor and read the output tensor manually, one sample at a time.
# This mirrors how it actually runs on the MCU later — no shortcuts.
interpreter = tf.lite.Interpreter(model_path="models/baseline_int8.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

# --- Understand the quantization parameters ---
# Because we forced int8 input/output during conversion, the model no longer
# accepts normalized float values (like -0.42, 1.83, ...) directly.
# It was calibrated to expect int8 values in range [-128, 127], where each
# int8 value maps back to a float value via:
#     real_value = (int8_value - zero_point) * scale
# So to go the OTHER way (float input -> int8 the model expects), we invert:
#     int8_value = round(real_value / scale) + zero_point
# `scale` and `zero_point` were decided by the converter during calibration
# in quantize.py (based on the range of the X_train data) — we don't pick
# them, we just read them off the model.
input_scale, input_zero_point = input_details["quantization"]
output_scale, output_zero_point = output_details["quantization"]

print(f"Input quant params:  scale={input_scale}, zero_point={input_zero_point}")
print(f"Output quant params: scale={output_scale}, zero_point={output_zero_point}")


def quantize_input(sample_float32):
    """Convert one normalized float32 feature vector to int8, per the formula above."""
    q = sample_float32 / input_scale + input_zero_point
    q = np.round(q)
    # Clip to int8's legal range in case rounding pushes slightly outside it.
    q = np.clip(q, -128, 127)
    return q.astype(np.int8)


def dequantize_output(output_int8):
    """Convert the model's int8 output scores back to float, so we can argmax them normally."""
    return (output_int8.astype(np.float32) - output_zero_point) * output_scale


y_pred = []
for i in range(len(X_test_norm)):
    sample = X_test_norm[i:i+1].astype(np.float32)
    sample_int8 = quantize_input(sample)

    # Feed the quantized sample into the interpreter's input buffer.
    interpreter.set_tensor(input_details["index"], sample_int8)
    # Run one forward pass.
    interpreter.invoke()
    # Read the raw int8 output (10 class scores, still quantized).
    output_int8 = interpreter.get_tensor(output_details["index"])
    output_float = dequantize_output(output_int8)

    y_pred.append(np.argmax(output_float))

y_pred = np.array(y_pred)

print("\n=== INT8 TFLite accuracy on load-1 test split ===")
print(classification_report(y_test, y_pred, target_names=label_names))
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))

# Direct comparison: how many predictions changed vs the float32 model?
# (Not "how many are wrong" — "how many DIFFER from float32's predictions",
# which isolates the effect of quantization specifically, separate from
# the model's original mistakes.)
