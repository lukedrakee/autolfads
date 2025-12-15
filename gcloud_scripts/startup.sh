#!/bin/bash
# Startup Script for PBT Client VMs
# Modernized for Docker with NVIDIA Container Toolkit

echo "Starting PBT client setup..."

# Install Docker if not present
if ! command -v docker &> /dev/null; then
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | apt-key add -
    add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
    apt-get update
    apt-get install -y docker-ce docker-ce-cli containerd.io
fi

# Install NVIDIA Container Toolkit (replaces nvidia-docker2)
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
apt-get update
apt-get install -y nvidia-container-toolkit
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker

# Install gcsfuse for bucket mounting
export GCSFUSE_REPO=gcsfuse-$(lsb_release -c -s)
echo "deb https://packages.cloud.google.com/apt $GCSFUSE_REPO main" | tee /etc/apt/sources.list.d/gcsfuse.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -
apt-get update
apt-get install -y gcsfuse

# Docker image will be pulled on demand by docker_setup.sh
# If using a custom image, build and push it first:
#   cd docker-build-steps
#   docker build -t gcr.io/YOUR_PROJECT/radical:latest .
#   docker push gcr.io/YOUR_PROJECT/radical:latest
# Then update pbt_script_multiVM.py with your image name

# Enable user_allow_other for FUSE mounts
sed -i 's/#user_allow_other/user_allow_other/' /etc/fuse.conf

echo "PBT client setup complete."
