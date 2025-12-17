"""
Data utilities for RADICaL calcium imaging analysis.

Functions for loading, preprocessing, and saving calcium imaging data.
"""

import os
import h5py
import numpy as np


def load_data(data_dir, train_file="train_data.h5", valid_file="valid_data.h5"):
    """
    Load training and validation data from HDF5 files.

    Expected format: HDF5 with 'train_data' dataset of shape (trials, time, neurons)

    Returns:
        train_data: np.array of shape (n_trials, n_timepoints, n_neurons)
        valid_data: np.array of shape (n_trials, n_timepoints, n_neurons)
    """
    train_path = os.path.join(data_dir, train_file)
    valid_path = os.path.join(data_dir, valid_file)

    with h5py.File(train_path, 'r') as f:
        train_data = f['train_data'][:]

    with h5py.File(valid_path, 'r') as f:
        valid_data = f['train_data'][:]  # Same key name by convention

    print(f"Loaded training data: {train_data.shape}")
    print(f"Loaded validation data: {valid_data.shape}")

    return train_data.astype(np.float32), valid_data.astype(np.float32)


def preprocess_calcium_dff(raw_fluorescence, baseline_window=(0, 20)):
    """
    Convert raw fluorescence to dF/F.

    Use this if you want to use OUTPUT_DIST = 'zi-gamma' (non-negative data).

    Args:
        raw_fluorescence: array of shape (trials, timepoints, neurons)
        baseline_window: tuple (start, end) indices for baseline period

    Returns:
        dF/F normalized data, same shape as input (non-negative)
    """
    start, end = baseline_window

    # Calculate baseline F0 for each trial and neuron
    F0 = raw_fluorescence[:, start:end, :].mean(axis=1, keepdims=True)

    # Avoid division by zero
    F0 = np.maximum(F0, 1e-8)

    # Calculate dF/F
    dF_F = (raw_fluorescence - F0) / F0

    # Clip extreme values
    dF_F = np.clip(dF_F, -1.0, 10.0)

    return dF_F.astype(np.float32)


def preprocess_calcium_zscore(raw_fluorescence, axis=None):
    """
    Z-score calcium data (subtract mean, divide by std).

    Use this if you want to use OUTPUT_DIST = 'gaussian' (allows negative values).

    Args:
        raw_fluorescence: array of shape (trials, timepoints, neurons)
        axis: axis to compute mean/std over. Default None = global z-score.
              Use axis=(0,1) for per-neuron z-scoring.

    Returns:
        Z-scored data, same shape as input (can have negative values)
    """
    mean = raw_fluorescence.mean(axis=axis, keepdims=True)
    std = raw_fluorescence.std(axis=axis, keepdims=True)
    std = np.maximum(std, 1e-8)  # Avoid division by zero

    z_scored = (raw_fluorescence - mean) / std

    return z_scored.astype(np.float32)


def validate_data_for_distribution(data, output_dist):
    """
    Check if data is compatible with the chosen output distribution.

    Args:
        data: numpy array of calcium data
        output_dist: 'zi-gamma' or 'gaussian'

    Returns:
        True if compatible, raises warning if not
    """
    has_negative = np.any(data < 0)
    min_val = data.min()
    max_val = data.max()
    mean_val = data.mean()

    print(f"Data statistics: min={min_val:.3f}, max={max_val:.3f}, mean={mean_val:.3f}")

    if output_dist == 'zi-gamma' and has_negative:
        print("WARNING: Data has negative values but using 'zi-gamma' distribution!")
        print("  zi-gamma expects non-negative data (raw or dF/F).")
        print("  Consider using OUTPUT_DIST = 'gaussian' for z-scored data.")
        return False

    if output_dist == 'gaussian':
        print("Using Gaussian distribution (appropriate for z-scored data).")

    if output_dist == 'zi-gamma':
        print("Using zi-gamma distribution (appropriate for dF/F data).")

    return True


def save_data(data, filepath, dataset_name="train_data"):
    """
    Save data to HDF5 format.

    Args:
        data: numpy array of shape (trials, timepoints, neurons)
        filepath: path to save HDF5 file
        dataset_name: name of dataset in HDF5 file
    """
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)

    with h5py.File(filepath, 'w') as f:
        f.create_dataset(dataset_name, data=data, dtype=np.float32)

    print(f"Saved data to {filepath}, shape: {data.shape}")


def create_example_data(n_trials=200, n_timepoints=100, n_neurons=50, output_dir="./data", zscore=True):
    """
    Create synthetic calcium-like data for testing the pipeline.

    This generates fake data with realistic properties - use only for testing!

    Args:
        n_trials: Number of trials
        n_timepoints: Timepoints per trial
        n_neurons: Number of neurons
        output_dir: Where to save the data
        zscore: If True, z-score the data (for Gaussian output).
                If False, keep as non-negative (for zi-gamma output).
    """
    os.makedirs(output_dir, exist_ok=True)

    print("Generating synthetic calcium imaging data...")

    # Generate latent dynamics (what RADICaL tries to recover)
    latent_dim = 5
    latent = np.zeros((n_trials, n_timepoints, latent_dim))

    for trial in range(n_trials):
        # Random oscillatory dynamics
        freqs = np.random.uniform(0.05, 0.2, latent_dim)
        phases = np.random.uniform(0, 2*np.pi, latent_dim)
        t = np.arange(n_timepoints)
        for d in range(latent_dim):
            latent[trial, :, d] = np.sin(2*np.pi*freqs[d]*t + phases[d])

    # Project to neurons through random weights
    weights = np.random.randn(latent_dim, n_neurons) * 0.5
    rates = np.exp(latent @ weights)  # Non-negative rates

    # Add noise (gamma-distributed to mimic calcium)
    noise_scale = 0.3
    data = rates + np.random.gamma(2, noise_scale, rates.shape)

    # Z-score if requested (default for use with Gaussian output)
    if zscore:
        data = preprocess_calcium_zscore(data, axis=(0, 1))  # Per-neuron z-score
        print("Data z-scored (use OUTPUT_DIST = 'gaussian')")
    else:
        print("Data kept as non-negative (use OUTPUT_DIST = 'zi-gamma')")

    # Split into train/valid
    split = int(0.8 * n_trials)
    train_data = data[:split]
    valid_data = data[split:]

    # Save
    save_data(train_data, os.path.join(output_dir, "train_data.h5"))
    save_data(valid_data, os.path.join(output_dir, "valid_data.h5"))

    print(f"Created example data in {output_dir}/")
    print(f"  Training: {train_data.shape}")
    print(f"  Validation: {valid_data.shape}")
    print(f"  Data range: [{train_data.min():.2f}, {train_data.max():.2f}]")

    return train_data, valid_data


if __name__ == "__main__":
    # Run this file directly to create example data
    create_example_data()
