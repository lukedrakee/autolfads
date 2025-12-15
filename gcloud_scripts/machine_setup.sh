#!/bin/bash
# Batch create multiple GPU-enabled client VMs for PBT
#
# Usage: sh machine_setup.sh <base-name> <num-machines> <zone> [gpu-type]
# Example: sh machine_setup.sh pbtclient 4 us-central1-a nvidia-tesla-t4
#
# Note: Use 'sh' to run this script in Google Cloud Shell (not './')

base_name=$1
num_machine=$2
zone=$3
gpu=${4:-"nvidia-tesla-t4"}  # Updated default from k80 to t4

if [ -z "$base_name" ] || [ -z "$num_machine" ] || [ -z "$zone" ]; then
    echo "Usage: sh machine_setup.sh <base-name> <num-machines> <zone> [gpu-type]"
    echo "Example: sh machine_setup.sh pbtclient 4 us-central1-a nvidia-tesla-t4"
    exit 1
fi

gpu_node_counter=0

while [ $gpu_node_counter -lt $num_machine ]
	do
	gpu_node_counter=$((gpu_node_counter+1))
	machine_name="$base_name$gpu_node_counter"
	echo "creating $machine_name"
	sh create_instance.sh $machine_name $zone $gpu
	done