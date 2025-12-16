# RADICaL for 2-Photon Calcium Imaging: GCP Multi-VM Setup

This guide explains how to set up RADICaL (LFADS for calcium imaging) on Google Cloud Platform with multiple GPU VMs for distributed training with Population Based Training (PBT).

---

## Overview

**What you're building:**
```
┌─────────────────────────────────────────────────────────────────┐
│                    Google Cloud Platform                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────┐          ┌─────────────────────────────┐  │
│  │   Server VM     │          │       Client VMs (GPUs)     │  │
│  │                 │          │                             │  │
│  │  - MongoDB      │◄────────►│  - Docker containers        │  │
│  │  - PBT Server   │  coords  │  - LFADS model training     │  │
│  │  - Coordinates  │          │  - One model per GPU        │  │
│  │    hyperparams  │          │                             │  │
│  └─────────────────┘          └─────────────────────────────┘  │
│           │                              │                      │
│           └──────────────┬───────────────┘                      │
│                          ▼                                      │
│           ┌─────────────────────────────┐                       │
│           │     GCS Bucket              │                       │
│           │  - Your calcium imaging data│                       │
│           │  - Model checkpoints        │                       │
│           │  - Training results         │                       │
│           └─────────────────────────────┘                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Components You Need

### 1. External Dependencies (NOT in this repo)

| Package | Source | Purpose |
|---------|--------|---------|
| **lfadslite** | [snel-repo/lfads-cd](https://github.com/snel-repo/lfads-cd) `radical` branch | The actual LFADS neural network with `zi-gamma` output distribution |
| **run_lfadslite.py** | Same repo | Helper functions for LFADS |

```bash
# Get the LFADS model code
git clone https://github.com/snel-repo/lfads-cd.git
cd lfads-cd
git checkout radical
```

### 2. From This Repository

| File | Purpose |
|------|---------|
| `radical_hyperparameters.py` | RADICaL-specific hyperparameter configuration |
| `pbt_opt/lfads_wrapper/lfads_wrapper.py` | Wrapper that calls lfadslite for training |
| `pbt_opt/lfads_wrapper/run_posterior_mean_sampling.py` | Extract latent factors from trained model |
| `pbt_opt/server.py` | PBT server (coordinates distributed training) |
| `pbt_opt/client.py` | PBT client (runs on GPU VMs) |
| `gcloud_scripts/*.sh` | GCP infrastructure scripts |

---

## Data Format Requirements

Your 2-photon calcium imaging data must be in HDF5 format:

```
your_bucket/data/
├── lfads_train_data.h5
├── lfads_valid_data.h5
└── lfads_test_data.h5  (optional)
```

Each HDF5 file should contain:
```python
# Required shape: (num_trials, num_timepoints, num_neurons)
train_data = h5_file['train_data']  # shape: (N_trials, T, N_neurons)

# Example: 500 trials, 100 timepoints, 200 neurons
# train_data.shape = (500, 100, 200)
```

**Data preprocessing tips for calcium imaging:**
- Use ΔF/F (normalized fluorescence change)
- Align trials to behavioral events
- Ensure consistent trial lengths
- Remove artifacts/bad trials

---

## RADICaL-Specific Configuration

The key difference from standard LFADS is the output distribution. In your training script:

```python
# CRITICAL: This makes it RADICaL (for calcium) vs LFADS (for spikes)
hyperparameters = {
    'output_dist': 'zi-gamma',  # Zero-inflated gamma for calcium

    # RADICaL-specific parameters
    'temporal_shift': 0,
    'fac_2_rates_transform': 'linscaledsigmoid',
    'gamma_prior': 20.0,        # Tune this! Range: 1-100
    's_min': 0.1,
    'l2_gamma_distance_scale': 1e-4,
}
```

**Why `zi-gamma`?**
- Calcium fluorescence is non-negative (unlike spike counts which are integers)
- Zero-inflated gamma models the heavy-tailed, non-negative distribution of ΔF/F
- Accounts for baseline fluorescence and indicator saturation effects

---

## GCP Setup Steps

### Step 1: Create GCS Bucket

```bash
# Create bucket for your data and results
gsutil mb gs://your-bucket-name

# Upload your data
gsutil -m cp -r ./your_data/* gs://your-bucket-name/data/
```

### Step 2: Create Server VM

The server runs MongoDB and coordinates PBT training:

```bash
cd gcloud_scripts
sh server_set_up.sh my-server us-central1-a
```

**What this creates:**
- Debian 12 VM with MongoDB 7.0
- Python 3 with required packages
- Network access for client coordination

### Step 3: Create GPU Client VMs

```bash
sh machine_setup.sh pbtclient 4 us-central1-a nvidia-tesla-t4
```

**What this creates:**
- 4 GPU VMs (pbtclient1, pbtclient2, pbtclient3, pbtclient4)
- Docker with NVIDIA Container Toolkit
- Clones repo and builds Docker image with lfadslite

### Step 4: Configure Training Script

On the server, edit `pbt_opt/pbt_script_multiVM.py`:

```python
# Your settings
bucket_name = 'your-bucket-name'
data_path = 'data'
run_path = 'runs'
name = 'my-calcium-experiment'

nprocess_gpu = 3  # Number of parallel models per GPU
```

### Step 5: Run Training

```bash
# SSH to server
gcloud compute ssh my-server --zone=us-central1-a

# Run PBT
cd ~/autolfads
python3 pbt_opt/pbt_script_multiVM.py
```

---

## Extracting Results (Post-Training)

After training completes, extract latent factors:

```python
from lfads_wrapper.run_posterior_mean_sampling import run_posterior_sample_and_average

# Run posterior sampling on best model
run_posterior_sample_and_average(
    model_dir='/path/to/best/model',
    data_file='lfads_valid_data.h5',
    output_file='posterior_means.h5'
)
```

**Output contains:**
- `factors` - Latent neural dynamics (shape: trials × time × factors_dim)
- `output_dist_params` - Inferred firing rates
- `controller_outputs` - If using controller

---

## Cost Considerations

| Resource | Approximate Cost |
|----------|-----------------|
| Server VM (n1-standard-4) | ~$0.15/hour |
| Client VM with T4 GPU | ~$0.35/hour each |
| 4 clients for 24 hours | ~$34 |

**Tips to reduce cost:**
- Use preemptible VMs (80% cheaper, but can be terminated)
- Start with fewer clients to test
- Delete VMs when not training

---

## Troubleshooting

### "Cannot find lfadslite"
Ensure the lfadslite package is in your PYTHONPATH:
```bash
export PYTHONPATH="/path/to/lfads-cd:$PYTHONPATH"
```

### "Docker container not starting"
Check if the Docker image built correctly:
```bash
gcloud compute ssh pbtclient1 --command="docker images | grep radical"
```

### "MongoDB connection refused"
Verify MongoDB is running:
```bash
gcloud compute ssh my-server --command="sudo systemctl status mongod"
```

---

## Files Summary

```
Required from this repo:
├── pbt_opt/
│   ├── server.py              # PBT coordination
│   ├── client.py              # Worker process
│   ├── lfads_wrapper/
│   │   ├── lfads_wrapper.py   # Training wrapper
│   │   └── run_posterior_mean_sampling.py  # Extract factors
│   └── pbt_script_multiVM.py  # Main config (edit this)
├── gcloud_scripts/            # Infrastructure setup
└── radical_2photon_core/
    └── radical_hyperparameters.py  # Reference config

Required external:
└── lfads-cd/ (radical branch)
    ├── lfadslite.py           # LFADS model with zi-gamma
    └── run_lfadslite.py       # Helper functions
```
