# RADICaL Core Files for 2-Photon Calcium Imaging

This folder contains the essential components for running RADICaL (Rate and Dynamics Inferred from Calcium) on 2-photon calcium imaging data.

---

## What is RADICaL?

**RADICaL** is an extension of LFADS (Latent Factor Analysis via Dynamical Systems) specifically designed for calcium imaging data.

| Standard LFADS | RADICaL |
|----------------|---------|
| Spike count data | Calcium fluorescence (ΔF/F) |
| Poisson output distribution | Zero-inflated gamma (`zi-gamma`) |
| Integer observations | Continuous, non-negative observations |

**The key difference is ONE parameter:** `output_dist = 'zi-gamma'`

---

## Files in This Folder

| File | Description |
|------|-------------|
| `radical_hyperparameters.py` | All RADICaL-specific hyperparameters with documentation |
| `SETUP_GCP_MULTIVM.md` | Guide for GCP distributed training setup |
| `SETUP_LOCAL_GPU.md` | Guide for local single-GPU training |
| `README.md` | This file |

---

## Quick Reference: RADICaL Parameters

```python
# CRITICAL - This makes LFADS work with calcium imaging
output_dist = 'zi-gamma'  # Zero-inflated gamma distribution

# Calcium-specific parameters
temporal_shift = 0                           # Indicator dynamics offset
fac_2_rates_transform = 'linscaledsigmoid'  # Factor to rate mapping
gamma_prior = 20.0                          # Shape of rate distribution
s_min = 0.1                                 # Minimum gamma scale
```

---

## External Dependencies

**You MUST get the LFADS model from a separate repository:**

### Option 1: TensorFlow 1.x (Original)
```bash
git clone https://github.com/snel-repo/lfads-cd.git
cd lfads-cd
git checkout radical  # Contains zi-gamma implementation
```

### Option 2: TensorFlow 2.x (Recommended for local)
```bash
git clone https://github.com/snel-repo/autolfads-tf2.git
cd autolfads-tf2
pip install -e lfads_tf2
```

---

## Data Format

Your calcium imaging data should be:
- **Format:** HDF5
- **Shape:** `(trials, timepoints, neurons)`
- **Values:** ΔF/F (normalized fluorescence change)

Example:
```python
import h5py
import numpy as np

# 500 trials, 100 timepoints, 200 neurons
data = np.random.randn(500, 100, 200).astype(np.float32)

with h5py.File('lfads_train_data.h5', 'w') as f:
    f.create_dataset('train_data', data=data)
```

---

## MATLAB Analysis Files

For post-training analysis, use these files from the parent repository:

| File | Purpose |
|------|---------|
| `+PBT_analysis/load_h5_data.m` | Load HDF5 model outputs |
| `+PBT_analysis/load_PMs_data.m` | Load posterior mean samples |
| `+PBT_analysis/plot_R2.m` | Compute R² between true and inferred rates |
| `+PBT_analysis/make_pbt_run_plots.m` | Generate result visualizations |

---

## Choosing Your Setup

| If you want... | Use |
|----------------|-----|
| **Simplest setup** | autolfads-tf2 locally → `SETUP_LOCAL_GPU.md` |
| **Original RADICaL code** | lfads-cd locally → `SETUP_LOCAL_GPU.md` |
| **Distributed training** | Full GCP setup → `SETUP_GCP_MULTIVM.md` |
| **Automatic hyperparameter optimization** | GCP with PBT → `SETUP_GCP_MULTIVM.md` |

---

## Citation

If you use RADICaL, please cite:

```bibtex
@inproceedings{schimel2021radical,
  title={RADICaL: A deep learning method for inferring neural dynamics from calcium imaging},
  author={Schimel, Marine and others},
  booktitle={NeurIPS},
  year={2021}
}

@article{pandarinath2018inferring,
  title={Inferring single-trial neural population dynamics using sequential auto-encoders},
  author={Pandarinath, Chethan and others},
  journal={Nature Methods},
  year={2018}
}
```

---

## Questions?

- **LFADS methodology:** See [Nature Methods paper](https://www.nature.com/articles/s41592-018-0109-9)
- **RADICaL paper:** See NeurIPS 2021
- **Code issues:** [snel-repo/lfads-cd](https://github.com/snel-repo/lfads-cd) or [snel-repo/autolfads-tf2](https://github.com/snel-repo/autolfads-tf2)
