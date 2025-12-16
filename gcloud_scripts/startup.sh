#!/bin/bash
# =============================================================================
# PBT Client VM Startup Script
# =============================================================================
# This script runs automatically when a client VM starts.
# It installs all dependencies and builds the Docker image locally.
#
# What this script does:
#   1. Installs Docker and NVIDIA Container Toolkit
#   2. Installs gcsfuse for GCS bucket mounting
#   3. Clones the autolfads repository
#   4. Builds the RADICaL Docker image locally
# =============================================================================

set -e  # Exit on any error

echo "=========================================="
echo "PBT Client VM Setup Starting..."
echo "=========================================="

# Configuration - modify these as needed
REPO_URL="https://github.com/lukedrakee/autolfads.git"
REPO_BRANCH="claude/radical-neural-analysis-01KpfvJqeYWzZEritRQ1wovr"
DOCKER_IMAGE_NAME="radical:latest"
INSTALL_DIR="/opt/autolfads"

# -----------------------------------------------------------------------------
# Step 1: Install Docker (if not present)
# -----------------------------------------------------------------------------
echo "[1/5] Checking Docker installation..."
if ! command -v docker &> /dev/null; then
    echo "Installing Docker..."
    apt-get update
    apt-get install -y apt-transport-https ca-certificates curl gnupg lsb-release

    # Add Docker's official GPG key
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

    # Add Docker repository
    echo "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null

    apt-get update
    apt-get install -y docker-ce docker-ce-cli containerd.io

    echo "Docker installed successfully."
else
    echo "Docker already installed."
fi

# -----------------------------------------------------------------------------
# Step 2: Install NVIDIA Container Toolkit
# -----------------------------------------------------------------------------
echo "[2/5] Setting up NVIDIA Container Toolkit..."
if ! dpkg -l | grep -q nvidia-container-toolkit; then
    distribution=$(. /etc/os-release; echo $ID$VERSION_ID)

    # Add NVIDIA GPG key and repository
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

    apt-get update
    apt-get install -y nvidia-container-toolkit

    # Configure Docker to use NVIDIA runtime
    nvidia-ctk runtime configure --runtime=docker
    systemctl restart docker

    echo "NVIDIA Container Toolkit installed successfully."
else
    echo "NVIDIA Container Toolkit already installed."
fi

# -----------------------------------------------------------------------------
# Step 3: Install gcsfuse for bucket mounting
# -----------------------------------------------------------------------------
echo "[3/5] Installing gcsfuse..."
if ! command -v gcsfuse &> /dev/null; then
    export GCSFUSE_REPO=gcsfuse-$(lsb_release -c -s)
    echo "deb https://packages.cloud.google.com/apt $GCSFUSE_REPO main" | tee /etc/apt/sources.list.d/gcsfuse.list
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -
    apt-get update
    apt-get install -y gcsfuse

    # Enable user_allow_other for FUSE mounts
    sed -i 's/#user_allow_other/user_allow_other/' /etc/fuse.conf

    echo "gcsfuse installed successfully."
else
    echo "gcsfuse already installed."
fi

# -----------------------------------------------------------------------------
# Step 4: Clone the repository
# -----------------------------------------------------------------------------
echo "[4/5] Cloning autolfads repository..."
apt-get install -y git

if [ -d "$INSTALL_DIR" ]; then
    echo "Repository already exists, pulling latest..."
    cd $INSTALL_DIR
    git fetch origin
    git checkout $REPO_BRANCH
    git pull origin $REPO_BRANCH
else
    git clone $REPO_URL $INSTALL_DIR
    cd $INSTALL_DIR
    git checkout $REPO_BRANCH
fi

echo "Repository ready at $INSTALL_DIR"

# -----------------------------------------------------------------------------
# Step 5: Build Docker image locally
# -----------------------------------------------------------------------------
echo "[5/5] Building RADICaL Docker image..."
cd $INSTALL_DIR/docker-build-steps

# Build the image
docker build -t $DOCKER_IMAGE_NAME .

echo "Docker image '$DOCKER_IMAGE_NAME' built successfully."

# Verify GPU access in Docker
echo "Verifying GPU access in Docker..."
docker run --rm --gpus all $DOCKER_IMAGE_NAME nvidia-smi || echo "Warning: GPU verification failed, but continuing..."

# -----------------------------------------------------------------------------
# Setup complete
# -----------------------------------------------------------------------------
echo "=========================================="
echo "PBT Client VM Setup Complete!"
echo "=========================================="
echo ""
echo "Docker image: $DOCKER_IMAGE_NAME"
echo "Repository: $INSTALL_DIR"
echo ""
echo "This VM is ready to receive PBT training jobs."
