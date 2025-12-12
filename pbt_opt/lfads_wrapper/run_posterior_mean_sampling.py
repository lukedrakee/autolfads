#!/usr/bin/env python3
"""
Posterior Mean Sampling Script for LFADS
Modernized for Python 3.9+ and TensorFlow 2.x
"""

from lfads_wrapper import lfadsWrapper
import json
import os
import glob
import collections.abc
import sys
import shutil
import tf_compat


def convert(data):
    """
    Convert JSON unicode to Python string (compatibility function).

    Args:
        data: Data to convert (can be string, mapping, or iterable)

    Returns:
        Converted data with proper string types
    """
    if isinstance(data, str):
        return data
    elif isinstance(data, bytes):
        return data.decode('utf-8')
    elif isinstance(data, collections.abc.Mapping):
        return dict(map(convert, data.items()))
    elif isinstance(data, collections.abc.Iterable):
        return type(data)(map(convert, data))
    else:
        return data


def get_hps(run_path, params_to_overwrite):
    """
    Load hyperparameters from file.

    Args:
        run_path: Path to run directory
        params_to_overwrite: Dictionary of parameters to override

    Returns:
        Hyperparameters dictionary
    """
    # look for hyperparams.txt file
    fdir = os.path.join(run_path, 'hyperparameters-*.txt')
    flist = tf_compat.glob(fdir)
    # fail if no files were found
    assert len(flist) != 0, "Directory " + fdir + " did not contain any hyperparameters-*.txt files"
    # otherwise, take the last one
    fname = flist[-1]

    print(f'Loading {fname} for posterior-mean-sampling.')

    with tf_compat.file_open(fname, 'r') as jfile:
        hps = json.load(jfile)

    # convert unicode to python string
    hps = convert(hps)
    # replace string true and false with python true and false
    for k, v in hps.items():
        if v in ['True', 'true']:
            v = True
        elif v in ['False', 'false']:
            v = False
        hps[k] = v

    for hp, val in params_to_overwrite.items():
        hps[hp] = val
    print(hps)

    return hps


def run_train(run_path, params_to_overwrite, num_steps, ckpt_load_path=None):
    """
    Run training from a saved checkpoint.

    Args:
        run_path: Path to run directory
        params_to_overwrite: Parameters to override
        num_steps: Number of training steps
        ckpt_load_path: Optional path to checkpoint to load
    """
    lfads = lfadsWrapper()
    if ckpt_load_path and run_path != ckpt_load_path:
        lfads.copy_checkpoint(ckpt_load_path, run_path, 'checkpoint', True)
        lfads.copy_checkpoint(ckpt_load_path, run_path, 'checkpoint_lve', False)
        # copy hyperparameter files
        print(ckpt_load_path)
        for file in glob.glob(os.path.join(ckpt_load_path, 'hyperparameters*')):
            shutil.copy(file, run_path)

    hps = get_hps(run_path, params_to_overwrite)
    hps['kind'] = 'train'
    lfads.train(hps, run_path, run_path, num_steps)


def write_model_params(run_path):
    """
    Write model parameters to file.

    Args:
        run_path: Path to run directory
    """
    hps = get_hps(run_path, {})
    hps['kind'] = 'write_model_params'
    lfads = lfadsWrapper()
    lfads.write_model_params(hps, run_path)


def run_posterior_sample_and_average(run_path, params_to_overwrite):
    """
    Run posterior sampling and averaging.

    Args:
        run_path: Path to run directory
        params_to_overwrite: Parameters to override
    """
    hps = get_hps(run_path, params_to_overwrite)
    hps['kind'] = 'posterior_sample_and_average'
    lfads = lfadsWrapper()
    lfads.posterior_mean_sample(hps, run_path)


if __name__ == "__main__":
    assert len(sys.argv) > 1, 'Must specify the lfadslite run directory.'
    run_path = sys.argv[1]

    if len(sys.argv) > 2:
        if sys.argv[2] == 'write_model_params':
            print('Writing model parameters.')
            write_model_params(run_path)
        elif sys.argv[2] == 'posterior_sample_and_average':
            if len(sys.argv) > 3:
                batch_size = int(sys.argv[3])
            else:
                batch_size = 100
                print(f'Running with default batch size of {batch_size}.')
            print('Running posterior sample and average.')
            run_posterior_sample_and_average(run_path, {'batch_size': batch_size})
        else:
            assert 0, 'Unknown run type!'
    else:
        batch_size = 100
        print(f'Running with default batch size of {batch_size}.')
        run_posterior_sample_and_average(run_path, {'batch_size': batch_size})
