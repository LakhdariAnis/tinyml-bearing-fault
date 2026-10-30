from scipy.io import loadmat

# loadmat reads a MATLAB .mat file and returns a Python dict.
# Every key is a variable name from the original MATLAB workspace;
# values are numpy arrays (or metadata for non-array entries).
data = loadmat("data/raw_3/IR021_3_217.mat")

for key, value in data.items():
    # Keys starting with "__" are MATLAB file metadata (version, header, etc),
    # not actual signal data — skip those to reduce noise.
    if key.startswith("__"):
        continue
    print(key, getattr(value, "shape", type(value)))
