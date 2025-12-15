# AutoLFADS/RADICaL Architecture Documentation

This document describes the architecture of the AutoLFADS/RADICaL codebase, separating the **neural network machine learning components** (specific to LFADS/RADICaL) from the **general infrastructure components** (applicable to other Google Cloud distributed training pipelines).

## Overview

The codebase implements Population Based Training (PBT) for hyperparameter optimization of LFADS (Latent Factor Analysis via Dynamical Systems) and RADICaL (Rate and Dynamics Inferred from Calcium Imaging), which are deep learning methods for analyzing neural population data.

---

## Directory Structure

```
autolfads/
├── pbt_opt/                    # Population Based Training optimization
│   ├── lfads_wrapper/          # Neural-specific LFADS model wrappers
│   └── run_scripts/            # Shell scripts to launch training
├── gcloud_scripts/             # GCP infrastructure scripts
├── tpu_scripts/                # TPU-specific scripts
├── docker-build-steps/         # Docker container configuration
├── run-manager-class/          # MATLAB run management
├── +PBT_analysis/              # MATLAB post-training analysis
└── utils/                      # MATLAB utilities
```

---

## Part 1: Neural-Specific Machine Learning Components

These files contain code unique to LFADS/RADICaL neural data analysis. You would need to keep or adapt these for calcium imaging analysis:

### Core Neural Model Wrappers

| File | Purpose |
|------|---------|
| `pbt_opt/lfads_wrapper/lfads_wrapper.py` | Main wrapper for LFADS model training, handles TF sessions, model building, checkpoint management |
| `pbt_opt/lfads_wrapper/lfads_wrapper_tfestimator.py` | Alternative TF Estimator-based wrapper for distributed training |
| `pbt_opt/lfads_wrapper/run_posterior_mean_sampling.py` | Post-training analysis: extracts latent factors, rates from trained model |

### Hyperparameter Configuration

| File | Purpose |
|------|---------|
| `pbt_opt/pbt_script_multiVM.py` | **KEY FILE** - Defines all LFADS/RADICaL hyperparameters for PBT search |

**Important hyperparameters defined here:**

```python
# RADICaL-specific parameters
svr.add_hp('output_dist', ['zi-gamma'])  # Zero-inflated gamma for calcium imaging
svr.add_hp('temporal_shift', [0])         # Temporal shift for deconvolution
svr.add_hp('fac_2_rates_transform', ['linscaledsigmoid'])
svr.add_hp('gamma_prior', ...)            # Gamma distribution prior for rates

# LFADS architecture parameters
svr.add_hp('factors_dim', [40])           # Latent factor dimensions
svr.add_hp('gen_dim', [64])               # Generator RNN hidden units
svr.add_hp('ic_dim', [64])                # Initial condition dimensions
svr.add_hp('co_dim', [2])                 # Controller output dimensions
```

### MATLAB Analysis Tools

| Directory | Purpose |
|-----------|---------|
| `+PBT_analysis/` | Post-training analysis: loading results, computing metrics, visualization |
| `utils/+Plot/`, `utils/+ExportFig/` | Plotting and figure export utilities |

### Key Neural Concepts

1. **LFADS (Latent Factor Analysis via Dynamical Systems)**
   - Infers latent neural dynamics from trial-averaged or single-trial data
   - Uses variational autoencoders with RNN dynamics
   - Outputs: latent factors, inferred rates, initial conditions

2. **RADICaL Extensions for Calcium Imaging**
   - Zero-inflated gamma output distribution for fluorescence data
   - Temporal shift handling for indicator dynamics
   - Specific loss functions for calcium deconvolution

---

## Part 2: General Infrastructure Components

These files implement distributed training infrastructure on GCP. They are **reusable for other ML projects**:

### Population Based Training (PBT) Framework

| File | Purpose | Reusability |
|------|---------|-------------|
| `pbt_opt/server.py` | **PBT Server**: Manages population of workers, exploit/explore operations, training loop | **Fully reusable** - generic hyperparameter optimization |
| `pbt_opt/client.py` | **PBT Client**: Runs on worker VMs, executes training jobs, reports results | **Fully reusable** - just plug in your train function |
| `pbt_opt/pbt_utils.py` | MongoDB database connection utilities | **Fully reusable** |
| `pbt_opt/pbt_helper_fn.py` | GCP instance discovery, zone/VM configuration | **Fully reusable** |

### How PBT Server/Client Architecture Works

```
┌─────────────────────────────────────────────────────────────────┐
│                        GCP Infrastructure                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────┐                     ┌─────────────────────────┐│
│  │ Server VM   │                     │      Client VMs         ││
│  │             │                     │                         ││
│  │  server.py  │◄────MongoDB────────►│  client.py (x N)       ││
│  │  - Population│                    │  - Train models         ││
│  │  - Exploit   │                    │  - Report performance   ││
│  │  - Explore   │                    │  - Load checkpoints     ││
│  │             │                     │                         ││
│  │  MongoDB    │                     │  Docker containers      ││
│  │  Instance   │                     │  with GPU               ││
│  └─────────────┘                     └─────────────────────────┘│
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                   GCS Bucket (shared storage)                ││
│  │    /data/           /run_path/          /checkpoints/        ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### GCP Infrastructure Scripts

| File | Purpose | Reusability |
|------|---------|-------------|
| `gcloud_scripts/server_set_up.sh` | Creates MongoDB server VM, installs dependencies | **Fully reusable** |
| `gcloud_scripts/create_instance.sh` | Creates GPU-enabled client VMs | **Fully reusable** |
| `gcloud_scripts/machine_setup.sh` | Batch creates multiple client VMs | **Fully reusable** |
| `gcloud_scripts/startup.sh` | VM startup: installs Docker, NVIDIA toolkit, gcsfuse | **Fully reusable** |
| `gcloud_scripts/docker_setup.sh` | Start/stop Docker containers on clients | **Fully reusable** |
| `gcloud_scripts/mounting.sh` | Mount/unmount GCS buckets via gcsfuse | **Fully reusable** |
| `gcloud_scripts/copy_logs.sh` | Copy training logs to shared storage | **Fully reusable** |
| `gcloud_scripts/start_stop_machine.sh` | Start/stop VMs | **Fully reusable** |
| `gcloud_scripts/update_docker_image.sh` | Update Docker image on all clients | **Fully reusable** |

### Docker Configuration

| File | Purpose |
|------|---------|
| `docker-build-steps/Dockerfile` | Container definition with TensorFlow 2.x, Python 3, dependencies |

---

## Part 3: Adapting for Your Own Project

### To use the PBT infrastructure with a different model:

1. **Modify the client's train function** (`pbt_opt/lfads_client.py`):
   ```python
   def obj_func(hps, run_save_path, ckpt_load_path, num_epoch):
       # Replace with your model training
       model = YourModel(hps)
       performance = model.train(...)
       return (performance, checkpoint_path)
   ```

2. **Define your hyperparameters** (like `pbt_script_multiVM.py`):
   ```python
   svr.add_hp('learning_rate', (0.0001, 0.01), explorable=True)
   svr.add_hp('hidden_dim', [64, 128, 256], explorable=True)
   # ... add your model-specific hyperparameters
   ```

3. **Keep infrastructure scripts unchanged**:
   - `server.py`, `client.py`, `pbt_utils.py` - generic PBT framework
   - `gcloud_scripts/` - GCP infrastructure setup
   - `docker-build-steps/Dockerfile` - modify dependencies only

### Key Classes to Understand

| Class | File | Purpose |
|-------|------|---------|
| `Server` | `server.py` | Main PBT server, orchestrates training |
| `Population` | `server.py` | Manages population of workers |
| `Worker` | `server.py` | Single member of population with hyperparameters |
| `Client` | `client.py` | Client process that executes training jobs |
| `DatabaseConnection` | `pbt_utils.py` | MongoDB interface for coordination |

---

## Part 4: Technology Stack

### Current (Modernized)

| Component | Version | Notes |
|-----------|---------|-------|
| Python | 3.9+ | All code updated for Python 3 |
| TensorFlow | 2.13+ | Using tf.compat.v1 for LFADS compatibility |
| MongoDB | 6.0+ | Server uses mongosh shell |
| Docker | Latest | NVIDIA Container Toolkit (--gpus all) |
| GCP Server Image | debian-12 | Standard Debian for MongoDB server (no GPU needed) |
| GCP Client Images | common-cu121 | Deep Learning VM with CUDA 12.1 for GPU workers |

### Dependencies (requirements.txt equivalent)

```
tensorflow>=2.13.0
tensorflow-probability>=0.20.0
pymongo>=4.0.0
google-cloud-storage>=2.0.0
grpcio>=1.50.0
protobuf>=4.21.0
numpy>=1.23.0
pandas>=1.5.0
h5py>=3.7.0
```

---

## Summary: What to Keep vs. What to Replace

### Keep (General Infrastructure)
- `pbt_opt/server.py` - PBT orchestration
- `pbt_opt/client.py` - Worker execution
- `pbt_opt/pbt_utils.py` - Database utilities
- `pbt_opt/pbt_helper_fn.py` - GCP helpers
- `gcloud_scripts/` - All infrastructure scripts
- `docker-build-steps/` - Container setup

### Replace/Modify (Neural-Specific)
- `pbt_opt/lfads_wrapper/` - Your model wrapper
- `pbt_opt/pbt_script_*.py` - Your hyperparameters
- `pbt_opt/lfads_client.py` - Your training function
- `+PBT_analysis/` - Your analysis tools

---

## Getting Started

> **Note:** When running shell scripts in Google Cloud Shell, use `sh script.sh` instead of `./script.sh` to avoid permission issues.

1. **Setup GCP Project**
   ```bash
   gcloud config set project YOUR_PROJECT_ID
   ```

2. **Create Server VM**
   ```bash
   cd gcloud_scripts
   sh server_set_up.sh my-server us-central1-a
   ```

3. **Create Client VMs**
   ```bash
   sh machine_setup.sh pbtclient 4 us-central1-a nvidia-tesla-t4
   ```

4. **Run PBT Training**
   ```bash
   # On server VM
   python pbt_opt/pbt_script_multiVM.py
   ```

See the original AutoLFADS documentation for detailed usage instructions.
