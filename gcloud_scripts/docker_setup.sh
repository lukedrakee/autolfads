#!/bin/sh
# Docker setup script for PBT clients
#
# Usage: sh docker_setup.sh <machine-tag> <container-name> <operation> [docker-image]
# Example: sh docker_setup.sh pbtclient docker_pbt start gcr.io/my-project/radical:latest

base_name=$1
docker_container=$2
operation=$3
docker_image=${4:-"tensorflow/tensorflow:2.13.0-gpu"}  # Default to public TensorFlow image
source="$HOME/bucket"

# Use --gpus all instead of --runtime=nvidia for modern NVIDIA Container Toolkit
cmd_docker="docker run --gpus all -it --mount src=$source,target=/bucket,type=bind -d --rm --name=$docker_container $docker_image"
cmd_stop="docker stop $docker_container"
check_docker="docker ps -q -f name=${docker_container}"
for instance in $(gcloud compute instances list --filter="tags:$base_name AND STATUS:RUNNING" --format="csv[no-heading](name)")

do
	zone=$(gcloud compute instances list --filter="name:$instance" --format="csv[no-heading](zone)")
	docker_status=$(gcloud compute ssh $instance --command="$check_docker" --zone=$zone)
	if [ $operation = "start" ]; then
		if [ ! $docker_status ]; then
		echo "Starting a new docker container - ${docker_container} , on ${instance}"
		gcloud compute ssh $instance --command="$cmd_docker" --zone=$zone
		else
		echo "WARNING: Docker Container ${docker_container} is already running on the client ${instance}"
		fi
	elif [ $operation = "stop" ]; then
                if [ $docker_status ]; then
		echo "Closing the pre-existing docker container on ${instance}"
		gcloud compute ssh $instance --command="$cmd_stop" --zone=$zone
		else
		echo "No docker container by name ${docker_container} running on the client ${instance}"
		fi
	fi
done
