#!/usr/bin/env python3
"""
LFADS Model Wrapper using TensorFlow Estimator API
Modernized for Python 3.9+ and TensorFlow 2.x
"""

import sys
import os
import re
import glob
import shutil
import warnings
import tf_compat

# Import lfadslite modules (these should be in PYTHONPATH)
from run_lfadslite import hps_dict_to_obj, jsonify_dict
from data_funcs import load_datasets
from helper_funcs import kind_dict, kind_dict_key
from lfadslite import LFADS
import data_funcs as utils

from subprocess import call
import time


class lfadsWrapper:
    """
    Wrapper class for LFADS model training using TF Estimator API.
    Handles training, checkpoint management, and posterior sampling.
    """

    def __init__(self):
        # Below variables are used to not reload the dataset
        self.datasets = None
        self.data_dir = None
        self.data_filename_stem = None

    def _copy_checkpoint(self, ckpt_load_path, target_path, ckpt_name):
        """
        Copy checkpoints (exploit operation) to new run directory.

        Args:
            ckpt_load_path: Path to checkpoint to copy
            target_path: Target directory path
            ckpt_name: Name of checkpoint file
        """
        # add / if doesn't exist
        target_path = os.path.join(target_path, '')
        ckpt_load_path = os.path.join(ckpt_load_path, '')
        # get the latest checkpoint files
        ckpt_file = os.path.join(ckpt_load_path, ckpt_name)

        with tf_compat.file_open(ckpt_file, 'r') as f:
            first_line = f.readline()
            f.seek(0)
            filedata = f.read()

        m = re.findall('"([^"]*)"', first_line)[0]
        # copy the checkpoints to the new directory
        m = os.path.join(ckpt_load_path, m + '*')

        if not tf_compat.file_exists(target_path):
            tf_compat.makedirs(target_path)
        else:
            tf_compat.rmtree(target_path)
            tf_compat.makedirs(target_path)
            warnings.warn("The directory for the new job already exists! Overwriting it!")

        assert call("gsutil -m cp %s %s" % (m, target_path), shell=True) == 0, "error copying checkpoints"

        ckpt_file_t = os.path.join(target_path, ckpt_name)
        # create LVE_CKPT_NAME at the target dir
        with tf_compat.file_open(ckpt_file_t, 'w') as the_file:
            the_file.write(filedata)

    def posterior_mean_sample(self, hps_dict, ckpt_load_path):
        """
        Run posterior mean sampling on the trained model.

        Args:
            hps_dict: Hyperparameters dictionary
            ckpt_load_path: Path to checkpoint
        """
        hps = hps_dict_to_obj(hps_dict)
        assert hps.kind == "posterior_sample_and_average", 'Kind is not posterior sample and average'
        # change the kind str to kind number
        hps.kind = kind_dict(hps.kind)
        hps.lfads_save_dir = ckpt_load_path
        self.load_datasets_if_necessary(hps)
        self.infer_dataset_properties(hps)
        model = self.build_model(hps, datasets=self.datasets)
        model_runs = model.write_model_runs(hps, self.datasets, None)
        return model_runs

    def train(self, hps_dict, lfads_save_path, ckpt_load_path, epochs_per_generation):
        """
        Train the LFADS model for one PBT generation.

        Args:
            hps_dict: Hyperparameters dictionary
            lfads_save_path: Path to save outputs
            ckpt_load_path: Path to load checkpoint from
            epochs_per_generation: Number of epochs to train

        Returns:
            Tuple of (reconstruction_cost, save_path)
        """
        hps = hps_dict_to_obj(hps_dict)
        assert hps.kind == "train", 'You can only invoke train with kind=train'
        hps.kind = kind_dict(hps.kind)
        if ckpt_load_path != '':
            if ckpt_load_path != lfads_save_path:
                # todo, allow different checkpoint name in lfadslite tf estimator
                # the default checkpoint name is used
                assert hps['checkpoint_pb_load_name'] == 'checkpoint', \
                    'custom checkpoint name is not implemented in lfadslite_tfestimator'
                ckpt_name = hps['checkpoint_pb_load_name']
                self._copy_checkpoint(ckpt_load_path, lfads_save_path, ckpt_name)
        else:
            # clear the run folder if it is a new run
            if tf_compat.file_exists(lfads_save_path):
                warnings.warn("The directory for the new job already exists! Overwriting it!")
                tf_compat.rmtree(lfads_save_path)

        hps.lfads_save_dir = lfads_save_path
        self.load_datasets_if_necessary(hps)
        self.infer_dataset_properties(hps)

        trial_recon_cost, samp_recon_cost = self._train(hps, self.datasets, epochs_per_generation)
        if hps.val_cost_for_pbt == 'heldout_samp':
            recon_cost = samp_recon_cost
        elif hps.val_cost_for_pbt == 'heldout_trial':
            recon_cost = trial_recon_cost
        else:
            assert 0, 'You must specify a heldout_samp or heldout_trial for recon cost!'
        return recon_cost, lfads_save_path

    def load_datasets_if_necessary(self, hps):
        """Load datasets only if they haven't been loaded or parameters changed."""
        stem_changed = self.data_filename_stem != hps.data_filename_stem
        data_dir_changed = self.data_dir != hps.data_dir
        if stem_changed or data_dir_changed:
            self.datasets = load_datasets(hps.data_dir, hps.data_filename_stem, hps)

    def infer_dataset_properties(self, hps):
        """Infer dataset names and dimensions from loaded files."""
        hps.dataset_names = []
        hps.dataset_dims = {}
        for key in self.datasets:
            hps.dataset_names.append(key)
            hps.dataset_dims[key] = self.datasets[key]['data_dim']
        # Python 3: use list() to get values from dict_values
        hps.num_steps = list(self.datasets.values())[0]['num_steps']
        hps.ndatasets = len(hps.dataset_names)

    def _train(self, hps, datasets, num_steps):
        """Internal training function."""
        model = self.build_model(hps, datasets=datasets)
        model.train_model(hps, run_mode='pbt', num_steps=num_steps)
        return model.trial_recon_cost, model.samp_recon_cost

    def build_model(self, hps, datasets=None):
        """Build the LFADS model."""
        model = LFADS(hps, datasets=datasets)

        fname = ''.join(('hyperparameters', '.txt'))
        hp_fname = os.path.join(hps.lfads_save_dir, fname)
        hps_for_saving = jsonify_dict(hps)
        utils.write_data(hp_fname, hps_for_saving, use_json=True)

        return model
