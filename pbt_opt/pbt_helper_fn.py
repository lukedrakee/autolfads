#!/usr/bin/env python3
"""
PBT Helper Functions for GCP Instance Discovery
Modernized for Python 3.9+
"""

import os
import subprocess


class pbtHelper:
    """
    Helper class for managing PBT on Google Cloud Platform.
    Handles discovery of VMs, zones, and configuration of compute resources.
    """

    def __init__(self, bucket_name, data_path, run_path, name, nprocess_gpu):
        """
        Initialize PBT helper with GCP configuration.

        Args:
            bucket_name: GCS bucket name
            data_path: Path to data within bucket
            run_path: Path for run outputs within bucket
            name: Name identifier for the run
            nprocess_gpu: Number of processes per GPU
        """
        self.bucket_name = bucket_name
        self.data_path = data_path
        self.run_path = run_path
        self.machine_name = 'pbtclient'
        self.container_name = 'docker_pbt'
        self.run_dir = "pbt_run"
        self.name = name
        self.nprocess_gpu = nprocess_gpu
        self.run_save_path = self.get_run_save_path(self.bucket_name, self.run_path)
        self.data_dir = self.get_data_dir(self.data_path)

        # Get server hostname
        subprocess.call(['export HOSTNAME'], shell=True)
        self.server_id = os.environ.get("HOSTNAME", "localhost")

        # Zone of the server VM
        try:
            zone_output = subprocess.check_output([
                'gcloud', 'compute', 'instances', 'list',
                '--filter', f'name=("{self.server_id}")',
                '--format', 'csv[no-heading](zone)'
            ], text=True)
            self.my_zone = zone_output.strip()
        except subprocess.CalledProcessError:
            self.my_zone = 'us-central1-a'  # Default zone

        # Get a list of names of VMs
        try:
            command = f"gcloud compute instances list --filter='tags:{self.machine_name} AND STATUS:RUNNING' --format='csv[no-heading](name)'"
            vms = subprocess.check_output(command, shell=True, text=True)
            self.vms_list = [v for v in vms.strip().split('\n') if v]
        except subprocess.CalledProcessError:
            self.vms_list = []

        # Get a list of the zones of the vms in vms_list
        try:
            command_zone = f"gcloud compute instances list --filter='tags:{self.machine_name} AND STATUS:RUNNING' --format='csv[no-heading](zone)'"
            vms_zone = subprocess.check_output(command_zone, shell=True, text=True)
            self.zone_list = [z for z in vms_zone.strip().split('\n') if z]
        except subprocess.CalledProcessError:
            self.zone_list = []

        self.vmzn = list(zip(self.vms_list, self.zone_list))
        self.computers = self.get_computers(self.vmzn)

        # Get server IP
        try:
            command_ip = f'gcloud --format="value(INTERNAL_IP)" compute instances list --filter=\'name:{self.server_id}\''
            server_ip = subprocess.check_output(command_ip, shell=True, text=True)
            self.server_ip = server_ip.strip().split('\n')[0]
        except subprocess.CalledProcessError:
            self.server_ip = '127.0.0.1'

    def get_run_save_path(self, bucket_name, run_path):
        """Generate GCS path for run outputs."""
        return f'gs://{bucket_name}/{run_path}/'

    def get_data_dir(self, data_path):
        """Generate mounted bucket path for data."""
        return f'/bucket/{data_path}/'

    def get_computers(self, vmzn):
        """
        Build computer configuration list for PBT.

        Args:
            vmzn: List of (vm_name, zone) tuples

        Returns:
            List of computer configuration dictionaries
        """
        computers = []
        keys = ['id', 'ip', 'max_processes', 'process_start_cmd', 'wait_for_process_start', 'zone']
        for vm, zn in vmzn:
            values = [vm, vm, self.nprocess_gpu, '/snel/autoLFADS-beta/pbt_opt/run_lfads_client.sh', True, zn]
            dictionary = dict(zip(keys, values))
            computers.append(dictionary)
        return computers
