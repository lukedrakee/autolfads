# AutoLFADS/RADICaL GCP Setup Guide

This guide walks you through setting up the complete PBT (Population Based Training) infrastructure on Google Cloud Platform for training LFADS/RADICaL models on neural data.

## Prerequisites

- A Google Cloud Platform account with billing enabled
- Access to Google Cloud Shell (or gcloud CLI installed locally)
- A GCS bucket for storing data and results

## Quick Start (5 Steps)

### Step 1: Clone the Repository

Open Google Cloud Shell and run:

```bash
# Clone the repository
git clone https://github.com/lukedrakee/autolfads.git
cd autolfads
git checkout claude/radical-neural-analysis-01KpfvJqeYWzZEritRQ1wovr

# Navigate to scripts directory
cd gcloud_scripts
```

### Step 2: Configure Your Settings

Edit `pbt_opt/pbt_script_multiVM.py` to set your bucket name:

```bash
# Open the file
nano ../pbt_opt/pbt_script_multiVM.py
```

Update these lines (around line 14-17):
```python
bucket_name = 'YOUR-BUCKET-NAME'    # Your GCS bucket
data_path = 'data'                   # Folder in bucket with your data
run_path = 'runs'                    # Folder for training outputs
name = 'my-experiment'               # Name for this training run
```

### Step 3: Create the Server VM

The server coordinates training across all client machines:

```bash
sh server_set_up.sh my-server us-central1-a
```

This will:
- Create a Debian 12 VM
- Install MongoDB 7.0
- Install Python 3 and required packages
- Configure MongoDB authentication

**Wait for the script to complete** (about 5-10 minutes).

### Step 4: Create Client VMs

Client VMs run the actual GPU training:

```bash
sh machine_setup.sh pbtclient 4 us-central1-a nvidia-tesla-t4
```

Arguments:
- `pbtclient` : Base name for VMs (creates pbtclient1, pbtclient2, etc.)
- `4` : Number of client VMs to create
- `us-central1-a` : GCP zone
- `nvidia-tesla-t4` : GPU type (optional, defaults to T4)

Each client VM will automatically:
- Install Docker and NVIDIA Container Toolkit
- Clone this repository
- Build the RADICaL Docker image locally

**Wait for all VMs to finish setup** (about 10-15 minutes).

You can check if the Docker image is ready on a client:
```bash
gcloud compute ssh pbtclient1 --zone=us-central1-a --command="docker images | grep radical"
```

### Step 5: Run Training

SSH into the server and start training:

```bash
# SSH to server
gcloud compute ssh my-server --zone=us-central1-a

# Navigate to repo and run
cd ~/autolfads
git pull  # Get latest code
python3 pbt_opt/pbt_script_multiVM.py
```

## Data Format

Your neural data should be in HDF5 format in your GCS bucket at `gs://YOUR-BUCKET/data/`:

```
data/
├── train_data.h5    # Training data
├── valid_data.h5    # Validation data
└── test_data.h5     # Test data (optional)
```

Each HDF5 file should contain:
- `train_data` : Array of shape (trials, time_steps, neurons)
- Additional fields as required by your specific LFADS configuration

## Monitoring

### Check Client VM Status
```bash
gcloud compute instances list --filter="tags:pbtclient"
```

### View Client Startup Logs
```bash
gcloud compute ssh pbtclient1 --zone=us-central1-a --command="sudo journalctl -u google-startup-scripts.service"
```

### Check Docker Container Status
```bash
gcloud compute ssh pbtclient1 --zone=us-central1-a --command="docker ps"
```

## Cleanup

To avoid ongoing charges, delete the VMs when done:

```bash
# Delete all client VMs
for i in 1 2 3 4; do
  gcloud compute instances delete pbtclient$i --zone=us-central1-a --quiet
done

# Delete server
gcloud compute instances delete my-server --zone=us-central1-a --quiet
```

## Troubleshooting

### "Docker image not found"
The startup script may still be running. Wait a few minutes and check:
```bash
gcloud compute ssh pbtclient1 --zone=us-central1-a --command="docker images"
```

### "MongoDB connection refused"
Ensure MongoDB is running on the server:
```bash
gcloud compute ssh my-server --zone=us-central1-a --command="sudo systemctl status mongod"
```

### "Permission denied" when running scripts
Use `sh` instead of `./`:
```bash
sh server_set_up.sh my-server us-central1-a
```

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Google Cloud Platform                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────┐          ┌─────────────────────────────┐  │
│  │   Server VM     │          │       Client VMs            │  │
│  │                 │          │                             │  │
│  │  - MongoDB      │◄────────►│  - Docker + GPU             │  │
│  │  - PBT Server   │          │  - LFADS Training           │  │
│  │  - Coordinates  │          │  - Multiple per VM          │  │
│  │    training     │          │                             │  │
│  └─────────────────┘          └─────────────────────────────┘  │
│           │                              │                      │
│           └──────────────┬───────────────┘                      │
│                          │                                      │
│                          ▼                                      │
│           ┌─────────────────────────────┐                       │
│           │     GCS Bucket              │                       │
│           │  - Training data            │                       │
│           │  - Model checkpoints        │                       │
│           │  - Results                  │                       │
│           └─────────────────────────────┘                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Support

For issues with this codebase, please open an issue on GitHub.

For questions about LFADS/RADICaL methodology, see the original publications:
- LFADS: Pandarinath et al., Nature Methods 2018
- RADICaL: Schimel et al., NeurIPS 2021
