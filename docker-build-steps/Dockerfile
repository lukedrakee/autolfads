# RADICaL/AutoLFADS Docker Image
# Modernized for TensorFlow 2.x and Python 3
FROM tensorflow/tensorflow:2.13.0-gpu

# Create directories
RUN mkdir -p /snel && mkdir -p /bucket

# Install system dependencies
RUN apt-get update && apt-get install -y \
    nmap \
    vim \
    tmux \
    nano \
    moreutils \
    git \
    && rm -rf /var/lib/apt/lists/*

# Clone the repository into the container
RUN git clone https://github.com/lukedrakee/autolfads.git /snel \
    && cd /snel \
    && git checkout claude/radical-neural-analysis-01KpfvJqeYWzZEritRQ1wovr

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install \
    pymongo>=4.0.0 \
    grpcio>=1.50.0 \
    protobuf>=4.21.0 \
    tensorflow-probability>=0.20.0 \
    google-cloud-storage>=2.0.0 \
    pandas>=1.5.0 \
    numpy>=1.23.0 \
    h5py>=3.7.0

# Set environment paths
ENV PATH="/snel/pbt_opt/:${PATH}"
ENV PYTHONPATH="/snel/pbt_opt/:/snel/lfadslite/:${PYTHONPATH}"

# Make scripts executable
RUN chmod +x /snel/pbt_opt/run_lfads_client.sh

EXPOSE 80
