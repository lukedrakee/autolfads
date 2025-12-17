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


def preprocess_calcium(raw_fluorescence, baseline_window=(0, 20)):
    """
    Convert raw fluorescence to dF/F for RADICaL input.

    Args:
        raw_fluorescence: array of shape (trials, timepoints, neurons)
        baseline_window: tuple (start, end) indices for baseline period

    Returns:
        dF/F normalized data, same shape as input
    """
    start, end = baseline_window

    # Calculate baseline F0 for each trial and neuron
    F0 = raw_fluorescence[:, start:end, :].mean(axis=1, keepdims=True)

    # Avoid division by zero
    F0 = np.maximum(F0, 1e-8)

    # Calculate dF/F
    dF_F = (raw_fluorescence - F0) / F0

    # Clip extreme values (optional but recommended)
    dF_F = np.clip(dF_F, -1.0, 10.0)

    return dF_F.astype(np.float32)


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


def create_example_data(n_trials=200, n_timepoints=100, n_neurons=50, output_dir="./data"):
    """
    Create synthetic calcium-like data for testing the pipeline.

    This generates fake data with realistic properties - use only for testing!
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

    return train_data, valid_data


if __name__ == "__main__":
    # Run this file directly to create example data
    create_example_data()
