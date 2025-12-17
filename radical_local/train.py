#!/usr/bin/env python
"""
RADICaL Training Script for 2-Photon Calcium Imaging

Usage:
    python train.py

Edit config.py to change settings.
"""

import os
import sys
import json
import time
import numpy as np

# Configuration
import config
from data_utils import load_data

# Try to import TensorFlow
try:
    import tensorflow as tf
    print(f"TensorFlow version: {tf.__version__}")
except ImportError:
    print("ERROR: TensorFlow not installed.")
    print("Install with: pip install tensorflow")
    sys.exit(1)


def build_lfads_model(input_shape, cfg):
    """
    Build a simplified LFADS-style model for calcium imaging.

    This is a minimal implementation. For full features, use:
    - autolfads-tf2: https://github.com/snel-repo/autolfads-tf2
    - lfads-cd: https://github.com/snel-repo/lfads-cd (radical branch)
    """
    n_timepoints, n_neurons = input_shape

    # Encoder
    inputs = tf.keras.Input(shape=(n_timepoints, n_neurons), name='calcium_input')

    # Bidirectional encoding
    encoded = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(cfg.GEN_DIM, return_sequences=True),
        name='encoder'
    )(inputs)

    encoded = tf.keras.layers.Dropout(1 - cfg.DROPOUT_KEEP)(encoded)

    # Initial condition (IC) encoding
    ic_mean = tf.keras.layers.Dense(cfg.IC_DIM, name='ic_mean')(encoded[:, 0, :])
    ic_logvar = tf.keras.layers.Dense(cfg.IC_DIM, name='ic_logvar')(encoded[:, 0, :])

    # Sample IC using reparameterization
    def sample_ic(args):
        mean, logvar = args
        std = tf.exp(0.5 * logvar)
        eps = tf.random.normal(tf.shape(std))
        return mean + eps * std

    ic = tf.keras.layers.Lambda(sample_ic, name='ic_sample')([ic_mean, ic_logvar])

    # Generator RNN
    gen_state = tf.keras.layers.Dense(cfg.GEN_DIM, activation='tanh', name='gen_init')(ic)

    # Run generator through time
    gen_cell = tf.keras.layers.GRUCell(cfg.GEN_DIM)
    gen_outputs = []
    state = gen_state

    for t in range(n_timepoints):
        output, [state] = gen_cell(tf.zeros((tf.shape(inputs)[0], 1)), [state])
        gen_outputs.append(output)

    gen_outputs = tf.stack(gen_outputs, axis=1)  # (batch, time, gen_dim)
    gen_outputs = tf.keras.layers.Dropout(1 - cfg.DROPOUT_KEEP)(gen_outputs)

    # Latent factors
    factors = tf.keras.layers.Dense(cfg.FACTORS_DIM, name='factors')(gen_outputs)

    # Output: predict calcium activity
    if cfg.OUTPUT_DIST == 'zi-gamma':
        # Gamma distribution for raw/dF/F data (non-negative)
        rate = tf.keras.layers.Dense(
            n_neurons,
            activation='softplus',
            name='gamma_rate'
        )(factors)

        concentration = tf.keras.layers.Dense(
            n_neurons,
            activation='softplus',
            name='gamma_concentration'
        )(factors) + 0.1  # Ensure positive

        outputs = [rate, concentration]

    elif cfg.OUTPUT_DIST == 'gaussian':
        # Gaussian distribution for z-scored data (can be negative)
        mean = tf.keras.layers.Dense(
            n_neurons,
            activation=None,  # No activation - can be negative
            name='gaussian_mean'
        )(factors)

        # Log-variance (learned)
        logvar = tf.keras.layers.Dense(
            n_neurons,
            activation=None,
            name='gaussian_logvar'
        )(factors)

        outputs = [mean, logvar]

    else:
        raise ValueError(f"Unknown output_dist: {cfg.OUTPUT_DIST}. Use 'gaussian' or 'zi-gamma'")

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='radical_lfads')

    return model, factors


def gamma_nll_loss(y_true, y_pred_rate, y_pred_conc, eps=1e-8):
    """
    Negative log-likelihood for gamma distribution.

    For zi-gamma (zero-inflated gamma), this approximates the calcium distribution.
    Use this for raw or dF/F data (non-negative values).
    """
    rate = y_pred_rate + eps
    concentration = y_pred_conc + eps

    nll = (
        concentration * tf.math.log(rate)
        - tf.math.lgamma(concentration)
        + (concentration - 1) * tf.math.log(y_true + eps)
        - rate * (y_true + eps)
    )

    return -tf.reduce_mean(nll)


def gaussian_nll_loss(y_true, y_pred_mean, y_pred_logvar):
    """
    Negative log-likelihood for Gaussian distribution.

    Use this for z-scored data (can have negative values).
    """
    # Gaussian NLL: 0.5 * (logvar + (y - mean)^2 / var)
    var = tf.exp(y_pred_logvar) + 1e-8
    nll = 0.5 * (y_pred_logvar + tf.square(y_true - y_pred_mean) / var)

    return tf.reduce_mean(nll)


class RADICaLTrainer:
    """Simple trainer for RADICaL model."""

    def __init__(self, model, cfg):
        self.model = model
        self.cfg = cfg
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=cfg.LEARNING_RATE)
        self.train_losses = []
        self.valid_losses = []

    def compute_loss(self, batch, outputs):
        """Compute loss based on output distribution type."""
        if self.cfg.OUTPUT_DIST == 'zi-gamma':
            rate, conc = outputs
            return gamma_nll_loss(batch, rate, conc)
        elif self.cfg.OUTPUT_DIST == 'gaussian':
            mean, logvar = outputs
            return gaussian_nll_loss(batch, mean, logvar)
        else:
            raise ValueError(f"Unknown output_dist: {self.cfg.OUTPUT_DIST}")

    @tf.function
    def train_step(self, batch):
        with tf.GradientTape() as tape:
            outputs = self.model(batch, training=True)
            loss = self.compute_loss(batch, outputs)

            # L2 regularization
            l2_loss = sum(tf.nn.l2_loss(w) for w in self.model.trainable_weights)
            total_loss = loss + self.cfg.L2_GEN * l2_loss

        gradients = tape.gradient(total_loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_weights))

        return loss

    def evaluate(self, data):
        outputs = self.model(data, training=False)
        loss = self.compute_loss(data, outputs)
        return float(loss)

    def train(self, train_data, valid_data, epochs):
        print(f"\nStarting training for {epochs} epochs...")
        print(f"Training data: {train_data.shape}")
        print(f"Batch size: {self.cfg.BATCH_SIZE}")
        print("-" * 50)

        n_batches = len(train_data) // self.cfg.BATCH_SIZE
        best_valid_loss = float('inf')

        for epoch in range(epochs):
            epoch_start = time.time()

            # Shuffle data
            indices = np.random.permutation(len(train_data))
            train_shuffled = train_data[indices]

            # Train batches
            epoch_loss = 0
            for i in range(n_batches):
                batch = train_shuffled[i*self.cfg.BATCH_SIZE:(i+1)*self.cfg.BATCH_SIZE]
                batch_loss = self.train_step(batch)
                epoch_loss += float(batch_loss)

            epoch_loss /= n_batches
            valid_loss = self.evaluate(valid_data)

            self.train_losses.append(epoch_loss)
            self.valid_losses.append(valid_loss)

            epoch_time = time.time() - epoch_start

            # Print progress
            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch:4d} | Train: {epoch_loss:.4f} | Valid: {valid_loss:.4f} | Time: {epoch_time:.1f}s")

            # Save best model
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                self.save_checkpoint(os.path.join(self.cfg.OUTPUT_DIR, "best_model"))

            # Periodic checkpoint
            if epoch % 100 == 0 and epoch > 0:
                self.save_checkpoint(os.path.join(self.cfg.OUTPUT_DIR, f"checkpoint_epoch_{epoch}"))

        print("-" * 50)
        print(f"Training complete. Best validation loss: {best_valid_loss:.4f}")

        # Save final model
        self.save_checkpoint(os.path.join(self.cfg.OUTPUT_DIR, "final_model"))

        return self.train_losses, self.valid_losses

    def save_checkpoint(self, path):
        os.makedirs(path, exist_ok=True)
        self.model.save_weights(os.path.join(path, "weights.h5"))

        # Save config
        cfg_dict = {k: v for k, v in vars(self.cfg).items() if not k.startswith('_')}
        with open(os.path.join(path, "config.json"), 'w') as f:
            json.dump(cfg_dict, f, indent=2, default=str)

        print(f"  Saved checkpoint to {path}")


def main():
    print("=" * 50)
    print("RADICaL Training for 2-Photon Calcium Imaging")
    print("=" * 50)

    # Create output directory
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # Load data
    print(f"\nLoading data from {config.DATA_DIR}...")
    train_data, valid_data = load_data(
        config.DATA_DIR,
        config.TRAIN_FILE,
        config.VALID_FILE
    )

    # Get data shape
    n_trials, n_timepoints, n_neurons = train_data.shape
    print(f"\nData shape: {n_trials} trials x {n_timepoints} timepoints x {n_neurons} neurons")

    # Build model
    print("\nBuilding model...")
    print(f"  Output distribution: {config.OUTPUT_DIST}")
    print(f"  Factors dim: {config.FACTORS_DIM}")

    model, _ = build_lfads_model((n_timepoints, n_neurons), config)
    model.summary()

    # Train
    trainer = RADICaLTrainer(model, config)
    train_losses, valid_losses = trainer.train(
        train_data,
        valid_data,
        config.MAX_EPOCHS
    )

    # Save loss curves
    np.savez(
        os.path.join(config.OUTPUT_DIR, "training_history.npz"),
        train_losses=train_losses,
        valid_losses=valid_losses
    )

    print(f"\nResults saved to {config.OUTPUT_DIR}/")
    print("Done!")


if __name__ == "__main__":
    main()
