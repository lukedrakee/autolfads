#!/bin/sh
# =============================================================================
# Docker Container Setup Script for PBT Clients
# =============================================================================
# Starts or stops Docker containers on PBT client VMs.
#
# Usage: sh docker_setup.sh <machine-tag> <container-name> <operation>
#
# Arguments:
#   machine-tag    : Tag used to identify client VMs (e.g., "pbtclient")
#   container-name : Name for the Docker container (e.g., "docker_pbt")
#   operation      : "start" or "stop"
#
# Example:
#   sh docker_setup.sh pbtclient docker_pbt start
#   sh docker_setup.sh pbtclient docker_pbt stop
# =============================================================================

base_name=$1
docker_container=$2
operation=$3

# Docker image built locally on each client VM by startup.sh
docker_image="radical:latest"

# Mount point for GCS bucket
source="$HOME/bucket"

if [ -z "$base_name" ] || [ -z "$docker_container" ] || [ -z "$operation" ]; then
    echo "Usage: sh docker_setup.sh <machine-tag> <container-name> <operation>"
    echo "Example: sh docker_setup.sh pbtclient docker_pbt start"
    exit 1
fi

# Docker commands
cmd_docker="docker run --gpus all -it --mount src=$source,target=/bucket,type=bind -d --rm --name=$docker_container $docker_image"
cmd_stop="docker stop $docker_container"
check_docker="docker ps -q -f name=${docker_container}"

# Find all running client VMs with the specified tag
for instance in $(gcloud compute instances list --filter="tags:$base_name AND STATUS:RUNNING" --format="csv[no-heading](name)")
do
    zone=$(gcloud compute instances list --filter="name:$instance" --format="csv[no-heading](zone)")

    docker_status=$(gcloud compute ssh $instance --command="$check_docker" --zone=$zone 2>/dev/null)

    if [ "$operation" = "start" ]; then
        if [ -z "$docker_status" ]; then
            echo "Starting Docker container '$docker_container' on $instance..."
            gcloud compute ssh $instance --command="$cmd_docker" --zone=$zone
        else
            echo "WARNING: Container '$docker_container' already running on $instance"
        fi
    elif [ "$operation" = "stop" ]; then
        if [ -n "$docker_status" ]; then
            echo "Stopping Docker container '$docker_container' on $instance..."
            gcloud compute ssh $instance --command="$cmd_stop" --zone=$zone
        else
            echo "No container '$docker_container' running on $instance"
        fi
    else
        echo "ERROR: Unknown operation '$operation'. Use 'start' or 'stop'."
        exit 1
    fi
done

echo "Docker setup complete."
