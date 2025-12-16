# RADICaL for 2-Photon Calcium Imaging: Local GPU Setup

This guide explains how to run RADICaL (LFADS for calcium imaging) locally on your own GPU without cloud infrastructure. This is simpler than the GCP setup and recommended for initial testing.

---

## Overview

**Local setup is simpler:**
```
┌─────────────────────────────────────────────────────────┐
│              Your Local Machine                          │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌─────────────────────────────────────────────────┐   │
│  │            Python Environment                    │   │
│  │                                                  │   │
│  │  ┌──────────────┐      ┌──────────────────┐    │   │
│  │  │ Your Script  │─────►│ lfadslite.LFADS  │    │   │
│  │  │ (config +    │      │ (neural network) │    │   │
│  │  │  training)   │      └──────────────────┘    │   │
│  │  └──────────────┘               │              │   │
│  │         │                       ▼              │   │
│  │         │              ┌──────────────────┐    │   │
│  │         └─────────────►│   Your GPU       │    │   │
│  │                        │   (Training)     │    │   │
│  │                        └──────────────────┘    │   │
│  └─────────────────────────────────────────────────┘   │
│                          │                              │
│                          ▼                              │
│           ┌─────────────────────────────┐              │
│           │     Local Disk              │              │
│           │  - Your calcium imaging data│              │
│           │  - Model checkpoints        │              │
│           │  - Training results         │              │
│           └─────────────────────────────┘              │
└─────────────────────────────────────────────────────────┘
```

**Advantages of local:**
- No cloud costs
- Simpler setup (no MongoDB, Docker, or multi-VM coordination)
- Easier debugging
- Good for initial experiments

**Disadvantages:**
- Slower than distributed training
- Limited to your GPU memory
- No automatic hyperparameter search (unless you implement it)

---

## Option A: Use autolfads-tf2 (Recommended - Easiest)

The SNEL lab has a **TensorFlow 2 version** that's designed for local use:

### Installation

```bash
# Clone the TF2 version
git clone https://github.com/snel-repo/autolfads-tf2.git
cd autolfads-tf2

# Create conda environment
conda create -n autolfads python=3.9
conda activate autolfads

# Install CUDA (adjust version for your GPU)
conda install cudatoolkit=11.8 cudnn=8.6 -c conda-forge

# Install the packages
pip install -e lfads_tf2
pip install -e tune_tf2  # For hyperparameter tuning
```

### Usage

```python
from lfads_tf2.models import LFADS
from lfads_tf2.utils import load_data

# Load your calcium imaging data
train_data, valid_data = load_data('path/to/your/data.h5')

# Configure for RADICaL (calcium imaging)
model = LFADS(
    # RADICaL-specific
    output_dist='zi-gamma',  # This is the key setting!

    # Architecture
    factors_dim=40,
    gen_dim=64,
    ic_dim=64,
    co_dim=2,

    # Data shape
    num_neurons=train_data.shape[-1],
    num_timesteps=train_data.shape[1],
)

# Train
model.fit(train_data, valid_data, epochs=500)

# Extract latent factors
factors = model.get_factors(valid_data)
```

---

## Option B: Use lfads-cd Directly (More Control)

If you need the original RADICaL implementation:

### Installation

```bash
# Clone lfads-cd with RADICaL
git clone https://github.com/snel-repo/lfads-cd.git
cd lfads-cd
git checkout radical

# Create environment
conda create -n radical python=3.7  # TF1 requires older Python
conda activate radical

# Install TensorFlow 1.x (required for this version)
pip install tensorflow-gpu==1.15.0

# Install other dependencies
pip install h5py numpy scipy matplotlib
```

### Create Training Script

Create `train_radical.py`:

```python
#!/usr/bin/env python
"""
Local training script for RADICaL (LFADS for calcium imaging)
"""
import os
import sys
import tensorflow as tf

# Add lfads-cd to path
sys.path.insert(0, '/path/to/lfads-cd')

from lfadslite import LFADS
from run_lfadslite import hps_dict_to_obj

# =============================================================================
# CONFIGURATION
# =============================================================================

# Paths
DATA_DIR = '/path/to/your/data'
OUTPUT_DIR = '/path/to/output'

# RADICaL hyperparameters (2-photon calcium imaging)
hps_dict = {
    # CRITICAL: This makes it RADICaL instead of standard LFADS
    'output_dist': 'zi-gamma',

    # RADICaL-specific parameters
    'temporal_shift': 0,
    'fac_2_rates_transform': 'linscaledsigmoid',
    'gamma_prior': 20.0,
    's_min': 0.1,
    'l2_gamma_distance_scale': 1e-4,
    'l2_fac_2_rates_scale': 1e-3,

    # Architecture
    'factors_dim': 40,
    'gen_dim': 64,
    'ic_dim': 64,
    'ic_enc_dim': 64,
    'co_dim': 2,
    'ci_enc_dim': 64,
    'con_dim': 64,

    # Training
    'learning_rate_init': 0.001,
    'batch_size': 64,  # Adjust based on GPU memory
    'keep_prob': 0.95,
    'keep_ratio': 0.5,
    'l2_gen_scale': 1e-3,
    'l2_con_scale': 1e-3,
    'kl_co_weight': 1e-5,
    'kl_ic_weight': 1e-5,

    # Data
    'data_dir': DATA_DIR,
    'data_filename_stem': 'lfads',

    # Training epochs
    'max_epochs': 500,
}

# =============================================================================
# TRAINING
# =============================================================================

def main():
    # Convert dict to hyperparameter object
    hps = hps_dict_to_obj(hps_dict)

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Build model
    with tf.Graph().as_default():
        model = LFADS(hps)

        # Training session
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            # Train
            for epoch in range(hps.max_epochs):
                train_cost = model.train_epoch(sess)
                valid_cost = model.validate(sess)

                if epoch % 10 == 0:
                    print(f"Epoch {epoch}: train={train_cost:.4f}, valid={valid_cost:.4f}")

                # Save checkpoint
                if epoch % 50 == 0:
                    model.save_checkpoint(sess, OUTPUT_DIR, epoch)

            # Save final model
            model.save_checkpoint(sess, OUTPUT_DIR, 'final')

    print(f"Training complete. Model saved to {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
```

---

## Data Format

Same format for both options - HDF5 files:

```python
import h5py
import numpy as np

# Your calcium imaging data
# Shape: (num_trials, num_timepoints, num_neurons)
calcium_data = np.random.randn(500, 100, 200)  # Example: 500 trials, 100 timepoints, 200 neurons

# Save as HDF5
with h5py.File('lfads_train_data.h5', 'w') as f:
    f.create_dataset('train_data', data=calcium_data)
```

**Preprocessing your calcium data:**

```python
import numpy as np

def preprocess_calcium(raw_fluorescence, baseline_window=(0, 20)):
    """
    Convert raw fluorescence to ΔF/F for LFADS input.

    Args:
        raw_fluorescence: shape (trials, timepoints, neurons)
        baseline_window: tuple of (start, end) indices for baseline

    Returns:
        dF/F normalized data
    """
    # Calculate baseline (F0)
    F0 = raw_fluorescence[:, baseline_window[0]:baseline_window[1], :].mean(axis=1, keepdims=True)

    # Calculate ΔF/F
    dF_F = (raw_fluorescence - F0) / (F0 + 1e-8)

    # Optional: clip extreme values
    dF_F = np.clip(dF_F, -1, 10)

    return dF_F.astype(np.float32)
```

---

## Extracting Latent Factors (Post-Training)

After training, extract the neural dynamics:

```python
def extract_factors(model_dir, data_file):
    """
    Extract latent factors from trained RADICaL model.
    """
    import h5py
    import tensorflow as tf

    # Load model and run posterior sampling
    # ... (depends on which framework you used)

    # Output will contain:
    # - factors: latent neural dynamics (trials × time × factors_dim)
    # - rates: inferred firing rates (trials × time × neurons)

    return factors, rates


# For autolfads-tf2:
from lfads_tf2.models import LFADS

model = LFADS.load('path/to/saved/model')
factors = model.get_factors(test_data)
rates = model.get_rates(test_data)
```

---

## Hyperparameter Tuning (Local)

Without PBT, you can do manual or grid search:

```python
from itertools import product

# Define search space
param_grid = {
    'factors_dim': [20, 40, 64],
    'gamma_prior': [10, 20, 50],
    'learning_rate_init': [0.0001, 0.001, 0.01],
}

# Grid search
best_score = float('inf')
best_params = None

for factors, gamma, lr in product(*param_grid.values()):
    params = {
        'factors_dim': factors,
        'gamma_prior': gamma,
        'learning_rate_init': lr,
        # ... other fixed params
    }

    # Train and evaluate
    score = train_and_evaluate(params)

    if score < best_score:
        best_score = score
        best_params = params
        print(f"New best: {params} -> {score}")
```

Or use Ray Tune / Optuna for more sophisticated search.

---

## GPU Memory Considerations

For local training, adjust batch size based on GPU memory:

| GPU | VRAM | Suggested batch_size |
|-----|------|---------------------|
| RTX 3060 | 12GB | 32-64 |
| RTX 3080 | 10GB | 32-48 |
| RTX 3090 | 24GB | 64-128 |
| A100 | 40GB | 128-256 |

If you get OOM errors:
```python
# Reduce batch size
hps_dict['batch_size'] = 32

# Or reduce model size
hps_dict['gen_dim'] = 32
hps_dict['factors_dim'] = 20
```

---

## Files Summary (Local Setup)

```
Minimal local setup:
├── lfads-cd/  (clone from GitHub, radical branch)
│   ├── lfadslite.py        # LFADS model with zi-gamma
│   └── run_lfadslite.py    # Helper functions
│
├── your_project/
│   ├── train_radical.py    # Your training script
│   ├── radical_hyperparameters.py  # Config from this repo
│   └── data/
│       ├── lfads_train_data.h5
│       └── lfads_valid_data.h5
│
└── outputs/
    ├── checkpoints/
    └── results/

OR (easier):

├── autolfads-tf2/  (clone from GitHub)
│   ├── lfads_tf2/          # TF2 LFADS package
│   └── tune_tf2/           # Hyperparameter tuning
│
└── your_project/
    ├── train.py            # Your training script
    └── data/
```

---

## Comparison: Local vs GCP

| Aspect | Local GPU | GCP Multi-VM |
|--------|-----------|--------------|
| **Setup time** | ~30 min | ~2 hours |
| **Cost** | Electricity only | $30-100/day |
| **Training speed** | Slower | 4-16x faster |
| **Hyperparameter search** | Manual/limited | Automatic PBT |
| **Best for** | Testing, small data | Production, large data |
| **Complexity** | Simple | Complex |

**Recommendation:** Start local to verify your data and model work, then scale to GCP for full training if needed.
