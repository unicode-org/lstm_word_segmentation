import os
import numpy as np
import tensorflow as tf  # Needed to handle TF tensors
import h5py

def convert_to_numpy(value):
    """
    Convert TensorFlow tensors/variables to NumPy arrays (float32).
    Ensures we remove any TensorFlow-specific data.
    """
    if isinstance(value, tf.Tensor) or isinstance(value, tf.Variable):
        return value.numpy().astype(np.float32)
    elif isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.number):
        return value.astype(np.float32)
    else:
        return None  # Ignore non-numeric data

def convert_weights(npy_path):
    """Convert `weights.npy` to a TensorFlow-free HDF5 format."""
    if not os.path.exists(npy_path):
        print(f"❌ Error: {npy_path} not found!")
        return
    
    h5_path = npy_path.replace(".npy", "_tf_free.h5")
    
    # Load the weights
    print(f"🔍 Loading {npy_path}...")
    weights = np.load(npy_path, allow_pickle=True)

    # Convert all elements to NumPy arrays (remove TensorFlow dtypes)
    converted_weights = [convert_to_numpy(w) for w in weights if convert_to_numpy(w) is not None]

    # Save to HDF5 format
    with h5py.File(h5_path, "w") as hf:
        for i, w in enumerate(converted_weights):
            hf.create_dataset(f"weight_{i}", data=w)

    print(f"✅ Converted: {npy_path} -> {h5_path}")

if __name__ == "__main__":
    # Search for all `weights.npy` files and convert them
    for root, _, files in os.walk("."):
        for file in files:
            if file == "weights.npy":
                convert_weights(os.path.join(root, file))

    print("🚀 All weight files converted successfully!")

