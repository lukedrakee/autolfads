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

**If your data is already z-scored in HDF5 format:** No preprocessing needed! Just:

1. Put your files in `data/` (or update paths in `config.py`)
2. Make sure shape is `(trials, timepoints, neurons)`
3. Run `python train.py`

The loader auto-detects the dataset key in your HDF5 file.

**If you need to create HDF5 files:**
```python
import h5py

# Your data: (trials, timepoints, neurons)
with h5py.File('data/train_data.h5', 'w') as f:
    f.create_dataset('data', data=your_train_array)

with h5py.File('data/valid_data.h5', 'w') as f:
    f.create_dataset('data', data=your_valid_array)
```

## Key Settings

In `config.py`:

```python
# Choose based on your data preprocessing:
OUTPUT_DIST = "gaussian"   # For z-scored data (can have negative values)
# OUTPUT_DIST = "zi-gamma" # For raw/dF/F data (non-negative only)

# Tune these based on your data
FACTORS_DIM = 40      # Latent dimensions (try 10-100)
BATCH_SIZE = 64       # Reduce if GPU OOM
MAX_EPOCHS = 500      # Training duration
```

## Z-Scored vs Raw Data

| Your Data | Use This Setting |
|-----------|-----------------|
| Z-scored (mean=0, std=1) | `OUTPUT_DIST = "gaussian"` |
| Raw fluorescence | `OUTPUT_DIST = "zi-gamma"` |
| dF/F (non-negative) | `OUTPUT_DIST = "zi-gamma"` |

**Important:** If your data is z-scored (has negative values), you MUST use `"gaussian"`. The `"zi-gamma"` distribution only works with non-negative data.

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
