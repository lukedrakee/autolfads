# RADICaL Multi-VM Training on GCP (Simplified)

## The Core Concept

Multi-VM training runs multiple LFADS models in parallel across several GPU machines. Each model tries different hyperparameters, and Population Based Training (PBT) keeps the best-performing ones.

```
You need:
1. A place to store data and results  →  GCS Bucket
2. Multiple GPUs to train models      →  GPU VMs
3. A way to coordinate which hyperparameters to try  →  Simple shared file or database
```

---

## Simplest Possible Setup

### Option A: Manual Multi-VM (No Orchestration)

Skip the complex PBT infrastructure entirely. Just run independent training jobs:

```bash
# 1. Create a bucket
gsutil mb gs://my-calcium-data

# 2. Upload your data (HDF5 format)
gsutil cp my_train_data.h5 gs://my-calcium-data/

# 3. Create GPU VMs (repeat for each)
gcloud compute instances create gpu-vm-1 \
    --zone=us-central1-a \
    --machine-type=n1-standard-4 \
    --accelerator=type=nvidia-tesla-t4,count=1 \
    --image-family=pytorch-latest-gpu \
    --image-project=deeplearning-platform-release \
    --maintenance-policy=TERMINATE

# 4. SSH in and install
gcloud compute ssh gpu-vm-1
pip install tensorflow h5py

# 5. Clone LFADS and run with different hyperparameters on each VM
git clone https://github.com/snel-repo/autolfads-tf2.git
# Edit config, set output_dist='zi-gamma', run training
```

**Pros:** Simple, no dependencies, easy to debug
**Cons:** Manual hyperparameter management, no automatic coordination

---

### Option B: Use Vertex AI (Google's Managed Service)

Let Google handle the infrastructure:

```bash
# Package your training code
# Submit hyperparameter tuning job
gcloud ai hp-tuning-jobs create \
    --region=us-central1 \
    --config=hptuning_config.yaml
```

**Pros:** Google manages VMs, automatic hyperparameter tuning
**Cons:** Requires learning Vertex AI, some setup overhead

---

## What Makes It "RADICaL"

Regardless of infrastructure, the only thing that makes LFADS work for calcium imaging is:

```python
output_dist = 'zi-gamma'  # Instead of 'poisson' for spike data
```

Everything else (PBT, Docker, MongoDB) is optimization infrastructure, not the core algorithm.

---

## Data Format

Your calcium data needs to be:
- HDF5 file
- Shape: `(num_trials, num_timepoints, num_neurons)`
- Values: ΔF/F (normalized fluorescence)

```python
import h5py
import numpy as np

# Example: save your data
with h5py.File('train_data.h5', 'w') as f:
    # calcium_data shape: (500 trials, 100 timepoints, 150 neurons)
    f.create_dataset('train_data', data=calcium_data)
```

---

## Recommendation

**If you're not familiar with Docker/MongoDB/distributed systems:**

Start with the **Local GPU Setup** (see `SETUP_LOCAL_GPU.md`). Get a single model training successfully first. The multi-VM setup is only worth it if:
- You have a lot of data
- You need to try many hyperparameter combinations
- Training time on one GPU is prohibitively long

For most 2-photon experiments, a single GPU is sufficient.

---

## If You Still Want Full PBT Multi-VM

The original repo's approach uses:
- **MongoDB** on a server VM to track hyperparameters
- **Docker** on client VMs to ensure consistent environments
- **gcsfuse** to mount cloud storage as a filesystem

If you want to use this, the key files are:
- `gcloud_scripts/server_set_up.sh` - Creates server with MongoDB
- `gcloud_scripts/machine_setup.sh` - Creates GPU clients
- `pbt_opt/pbt_script_multiVM.py` - Main training script (edit bucket name, paths)

But expect debugging. The simpler approaches above are more reliable.
