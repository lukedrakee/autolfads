#!/usr/bin/env python
"""
Extract Latent Factors from Trained RADICaL Model

Usage:
    python extract_factors.py --model outputs/best_model --data data/valid_data.h5

This extracts the learned neural dynamics (latent factors) from your trained model.
"""

import os
import sys
import argparse
import json
import numpy as np
import h5py

import tensorflow as tf

# Import the model builder from train.py
from train import build_lfads_model


def load_model(model_dir):
    """Load trained model from checkpoint directory."""

    # Load config
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, 'r') as f:
        cfg_dict = json.load(f)

    # Create a simple config object
    class Config:
        pass

    cfg = Config()
    for k, v in cfg_dict.items():
        setattr(cfg, k, v)

    return cfg, model_dir


def extract_factors(model_dir, data_path, output_path=None):
    """
    Extract latent factors from data using trained model.

    Args:
        model_dir: Path to saved model checkpoint
        data_path: Path to HDF5 data file
        output_path: Where to save extracted factors (default: same dir as data)

    Returns:
        factors: numpy array of shape (n_trials, n_timepoints, factors_dim)
    """
    print(f"Loading model from {model_dir}...")
    cfg, _ = load_model(model_dir)

    # Load data
    print(f"Loading data from {data_path}...")
    with h5py.File(data_path, 'r') as f:
        data = f['train_data'][:]
    print(f"Data shape: {data.shape}")

    n_trials, n_timepoints, n_neurons = data.shape

    # Rebuild model architecture
    print("Building model...")
    model, factors_layer = build_lfads_model((n_timepoints, n_neurons), cfg)

    # Load weights
    weights_path = os.path.join(model_dir, "weights.h5")
    model.load_weights(weights_path)
    print("Loaded weights.")

    # Create a model that outputs factors
    # We need to extract the factors from an intermediate layer
    factor_model = tf.keras.Model(
        inputs=model.input,
        outputs=model.get_layer('factors').output
    )

    # Extract factors
    print("Extracting latent factors...")
    factors = factor_model.predict(data, batch_size=32, verbose=1)
    print(f"Factors shape: {factors.shape}")

    # Also get predictions
    predictions = model.predict(data, batch_size=32, verbose=1)

    # Save results
    if output_path is None:
        output_path = os.path.join(os.path.dirname(data_path), "extracted_factors.h5")

    print(f"Saving to {output_path}...")
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('factors', data=factors)
        f.create_dataset('input_data', data=data)

        if isinstance(predictions, list):
            f.create_dataset('predicted_rate', data=predictions[0])
            f.create_dataset('predicted_concentration', data=predictions[1])
        else:
            f.create_dataset('predictions', data=predictions)

        # Save metadata
        f.attrs['model_dir'] = model_dir
        f.attrs['data_path'] = data_path
        f.attrs['factors_dim'] = cfg.FACTORS_DIM

    print(f"\nDone! Factors saved to {output_path}")
    print(f"  - factors: {factors.shape}")
    print(f"  - Use these for downstream analysis (decoding, visualization, etc.)")

    return factors


def main():
    parser = argparse.ArgumentParser(
        description="Extract latent factors from trained RADICaL model"
    )
    parser.add_argument(
        "--model", "-m",
        default="outputs/best_model",
        help="Path to trained model directory"
    )
    parser.add_argument(
        "--data", "-d",
        default="data/valid_data.h5",
        help="Path to data HDF5 file"
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output path for extracted factors"
    )

    args = parser.parse_args()

    # Check paths exist
    if not os.path.exists(args.model):
        print(f"ERROR: Model directory not found: {args.model}")
        sys.exit(1)

    if not os.path.exists(args.data):
        print(f"ERROR: Data file not found: {args.data}")
        sys.exit(1)

    factors = extract_factors(args.model, args.data, args.output)

    return factors


if __name__ == "__main__":
    main()
