#!/usr/bin/env bash
# Create GCP Compute Instance with GPU for PBT Client
# Modernized for current GCP APIs and TensorFlow 2.x

INSTANCE_NAME=$1
ZONE=$2
GPU=${3:-"nvidia-tesla-t4"}  # Default to T4 (more cost-effective than K80)
NUM_GPUS=${4:-1}

gcloud compute instances create ${INSTANCE_NAME} \
  --zone ${ZONE} \
  --machine-type "n1-standard-4" \
  --subnet "default" \
  --maintenance-policy "TERMINATE" \
  --scopes "https://www.googleapis.com/auth/cloud-platform" \
  --accelerator type=${GPU},count=${NUM_GPUS} \
  --min-cpu-platform "Automatic" \
  --tags "pbtclient" \
  --image-family "common-cu121" \
  --image-project "deeplearning-platform-release" \
  --metadata="install-nvidia-driver=True" \
  --boot-disk-type "pd-ssd" \
  --boot-disk-size "100GB" \
  --boot-disk-device-name "${INSTANCE_NAME}-disk" \
  --metadata-from-file startup-script=./startup.sh
