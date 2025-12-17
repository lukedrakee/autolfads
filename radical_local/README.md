# RADICaL Local Setup

A minimal, self-contained implementation for running RADICaL (LFADS for 2-photon calcium imaging) on a local GPU.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Create example data (for testing)
python data_utils.py

# 3. Train
python train.py

# 4. Extract latent factors
python extract_factors.py --model outputs/best_model --data data/valid_data.h5
```

## Files

| File | Purpose |
|------|---------|
| `config.py` | All settings - **edit this** |
| `train.py` | Training script |
| `extract_factors.py` | Extract latent factors after training |
| `data_utils.py` | Data loading and preprocessing |
| `requirements.txt` | Python dependencies |

## Using Your Own Data

1. Format your calcium data as HDF5:
   ```python
   import h5py
   import numpy as np

   # Your data: (trials, timepoints, neurons)
   calcium_data = ...  # Shape: (500, 100, 150)

   with h5py.File('data/train_data.h5', 'w') as f:
       f.create_dataset('train_data', data=calcium_data)
   ```

2. Edit `config.py`:
   ```python
   DATA_DIR = "./data"
   TRAIN_FILE = "train_data.h5"
   VALID_FILE = "valid_data.h5"
   ```

3. Run training:
   ```bash
   python train.py
   ```

## Key Settings

In `config.py`:

```python
# What makes it RADICaL (for calcium, not spikes)
OUTPUT_DIST = "zi-gamma"  # Do not change

# Tune these based on your data
FACTORS_DIM = 40      # Latent dimensions (try 10-100)
GAMMA_PRIOR = 20.0    # Gamma prior (try 1-100)
BATCH_SIZE = 64       # Reduce if GPU OOM
MAX_EPOCHS = 500      # Training duration
```

## GPU Memory

If you get out-of-memory errors, reduce in `config.py`:
```python
BATCH_SIZE = 32    # or smaller
FACTORS_DIM = 20   # or smaller
GEN_DIM = 32       # or smaller
```

## Output

After training:
- `outputs/best_model/` - Best checkpoint
- `outputs/training_history.npz` - Loss curves
- `extracted_factors.h5` - Latent dynamics (after running extract_factors.py)

## What Are Latent Factors?

The extracted factors represent low-dimensional neural dynamics underlying your calcium recordings. Use them for:
- Decoding behavior
- Comparing conditions
- Visualization (PCA, UMAP)
- Identifying neural states

## Full LFADS Implementation

This is a simplified implementation. For the complete version with all features:
- **TensorFlow 2**: https://github.com/snel-repo/autolfads-tf2
- **TensorFlow 1**: https://github.com/snel-repo/lfads-cd (radical branch)
