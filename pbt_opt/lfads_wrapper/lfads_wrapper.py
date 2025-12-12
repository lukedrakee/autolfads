#!/usr/bin/env python3
"""
LFADS Model Wrapper for Population Based Training
Modernized for Python 3.9+ and TensorFlow 2.x
"""

import sys
import os
import re
import glob
import shutil
import ntpath
import warnings
import tf_compat

# Import lfadslite modules (these should be in PYTHONPATH)
from run_lfadslite import hps_dict_to_obj, jsonify_dict
from data_funcs import load_datasets
from helper_funcs import kind_dict, kind_dict_key
from lfadslite import LFADS
import data_funcs as utils

import tensorflow as tf

# TF2 compatibility
if hasattr(tf, 'compat') and hasattr(tf.compat, 'v1'):
    tf_v1 = tf.compat.v1
    tf_v1.disable_eager_execution()
else:
    tf_v1 = tf

allow_gpu_growth = True


class lfadsWrapper:
    """
    Wrapper class for LFADS model training in PBT framework.
    Handles training, checkpoint management, and posterior sampling.
    """

    def __init__(self):
        # Below variables are used to not reload the dataset
        self.datasets = None
        self.data_dir = None
        self.data_filename_stem = None

    def copy_hps(self, source_path, target_path):
        """
        Copy hyperparameter files to a new directory.

        Args:
            source_path: Source directory path
            target_path: Target directory path
        """
        # validate source path
        assert os.path.exists(source_path), "Can't find path " + source_path
        # validate target path
        assert os.path.exists(target_path), "Can't find path " + target_path
        # look for HP files
        fdir = os.path.join(source_path, 'hyperparameters-*.txt')
        flist = glob.glob(fdir)
        # fail if no files were found
        assert len(flist) != 0, "Directory " + fdir + " did not contain any hyperparameters-*.txt files"

        for f in flist:
            path, filename = os.path.split(f)
            fout = os.path.join(target_path, filename)
            print("Copying " + f + " to " + fout)
            tf_compat.copy(f, fout)

    def copy_checkpoint(self, ckpt_load_path, target_path, ckpt_name, overwrite_target_path=False):
        """
        Copy checkpoints (exploit operation) to new run directory.

        Args:
            ckpt_load_path: Path to checkpoint to copy
            target_path: Target directory path
            ckpt_name: Name of checkpoint file
            overwrite_target_path: Whether to overwrite existing target directory
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
        m = os.path.join(ckpt_load_path, m)
        m = m + '*'

        if not os.path.exists(target_path):
            os.makedirs(target_path)
        else:
            warnings.warn("The directory for the new job already exists!")
            if overwrite_target_path:
                tf_compat.rmtree(target_path)
                tf_compat.makedirs(target_path)
                warnings.warn("Overwriting it!")

        for file in tf_compat.glob(m):
            head, base = ntpath.split(file)
            target_file_path = os.path.join(target_path, base)
            tf_compat.copy(file, target_file_path)

        # replace the old ckpt path with the new one
        filedata = filedata.replace(ckpt_load_path, target_path)

        ckpt_file_t = os.path.join(target_path, ckpt_name)

        with tf_compat.file_open(ckpt_file_t, 'w') as the_file:
            the_file.write(filedata)

    def write_model_params(self, hps_dict, ckpt_load_path):
        """
        Write model parameters to file.

        Args:
            hps_dict: Hyperparameters dictionary
            ckpt_load_path: Path to checkpoint
        """
        hps = hps_dict_to_obj(hps_dict)
        assert hps.kind == "write_model_params", 'Kind is not write_model_params'
        # change the kind str to kind number
        hps.kind = kind_dict(hps.kind)
        hps.lfads_save_dir = ckpt_load_path
        self.load_datasets_if_necessary(hps)
        self.infer_dataset_properties(hps)
        fname = os.path.join(hps.lfads_save_dir, "model_params")
        tf_v1.reset_default_graph()
        config = tf_v1.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        config.gpu_options.allow_growth = allow_gpu_growth
        sess = tf_v1.Session(config=config)
        try:
            with sess.as_default():
                with tf.device(hps.device):
                    model = self.build_model(hps, datasets=self.datasets)
                    model_params = model.eval_model_parameters(use_nested=False,
                                                               include_strs="LFADS")
                    utils.write_data(fname, model_params, compression=None)
        except Exception:
            sess.close()
            tf_v1.reset_default_graph()
            raise

        return None

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

        tf_v1.reset_default_graph()
        config = tf_v1.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        config.gpu_options.allow_growth = allow_gpu_growth
        sess = tf_v1.Session(config=config)
        try:
            with sess.as_default():
                with tf.device(hps.device):
                    model_runs = self.write_model_runs(hps, self.datasets)
        except Exception:
            sess.close()
            tf_v1.reset_default_graph()
            raise

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
                ckpt_name = hps['checkpoint_pb_load_name']
                self.copy_checkpoint(ckpt_load_path, lfads_save_path, ckpt_name, True)
        else:
            # clear the run folder if it is a new run
            if os.path.exists(lfads_save_path):
                warnings.warn("The directory for the new job already exists! Overwriting it!")
                tf_compat.rmtree(lfads_save_path)

        hps.lfads_save_dir = lfads_save_path
        self.load_datasets_if_necessary(hps)
        self.infer_dataset_properties(hps)

        tf_v1.reset_default_graph()
        config = tf_v1.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        config.gpu_options.allow_growth = allow_gpu_growth
        sess = tf_v1.Session(config=config)
        try:
            with sess.as_default():
                with tf.device(hps.device):
                    trial_recon_cost, samp_recon_cost = self._train(hps, self.datasets, epochs_per_generation)
        except Exception:
            sess.close()
            tf_v1.reset_default_graph()
            raise

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

    def write_model_runs(self, hps, datasets, output_fname=None):
        """
        Run the model on the data and save the computed values.

        Args:
            hps: Hyperparameters object
            datasets: Dictionary of datasets
            output_fname: Optional output filename
        """
        model = self.build_model(hps, datasets=datasets)
        model.write_model_runs(datasets, output_fname)

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
        if hps.do_reset_learning_rate:
            sess = tf_v1.get_default_session()
            sess.run(model.learning_rate.initializer)

        model.train_model(datasets, num_steps)
        return model.trial_recon_cost, model.samp_recon_cost

    def build_model(self, hps, datasets=None, reuse=None):
        """Build the LFADS model."""
        with tf_v1.variable_scope("LFADS", reuse=reuse):
            model = LFADS(hps, datasets=datasets)

        if not os.path.exists(hps.lfads_save_dir):
            print("Save directory %s does not exist, creating it." % hps.lfads_save_dir)
            os.makedirs(hps.lfads_save_dir)

        cp_pb_ln = hps.checkpoint_pb_load_name
        cp_pb_ln = 'checkpoint' if cp_pb_ln == "" else cp_pb_ln
        if cp_pb_ln == 'checkpoint':
            print("Loading latest training checkpoint in: ", hps.lfads_save_dir)
            saver = model.seso_saver
        elif cp_pb_ln == 'checkpoint_lve':
            print("Loading lowest validation checkpoint in: ", hps.lfads_save_dir)
            saver = model.lve_saver
        else:
            print("Loading checkpoint: ", cp_pb_ln, ", in: ", hps.lfads_save_dir)
            saver = model.seso_saver

        ckpt = tf.train.get_checkpoint_state(hps.lfads_save_dir,
                                             latest_filename=cp_pb_ln)

        session = tf_v1.get_default_session()
        print("ckpt: ", ckpt)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            saver.restore(session, ckpt.model_checkpoint_path)
            # re-initialize the learning rate
            tf_v1.variables_initializer([model.learning_rate]).run()
        else:
            print("Created model with fresh parameters.")
            tf_v1.global_variables_initializer().run()

        if ckpt:
            train_step_str = re.search('-[0-9]+$', ckpt.model_checkpoint_path).group()
        else:
            train_step_str = '-0'

        fname = 'hyperparameters' + train_step_str + '.txt'
        hp_fname = os.path.join(hps.lfads_save_dir, fname)
        hps_for_saving = jsonify_dict(hps)
        utils.write_data(hp_fname, hps_for_saving, use_json=True)

        return model
