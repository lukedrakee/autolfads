#!/usr/bin/env python3
"""
Script for gathering and processing PBT run results.
Modernized for Python 3.9+

Example usage:
    python script_results.py --topdir /path/to/runs/PBT/lorenz_spike/test_pbt/ --hplist learning_rate_init
"""

import csv
import os
import sys
import glob
import pandas as pd
import argparse


def gather_run_csv(path, hp_names):
    """
    Gather performance and hyperparameter history from PBT run.

    Args:
        path: Path to PBT run directory
        hp_names: List of hyperparameter names to extract
    """
    outpath = path
    # get performance history for each worker and concatenate into a single file
    flist = list(glob.iglob(os.path.join(path, '*/perf_history.csv')))
    dnames = [os.path.basename(os.path.split(file)[0]) for file in flist]
    # unique generators
    gnames = list(set([get_digits(d.split('_')[0]) for d in dnames]))
    gnames.sort(key=int)

    if not gnames:
        print(f"No generation directories found in {path}")
        return

    flist = list(glob.iglob(os.path.join(path, 'g' + gnames[-1] + '_w*/perf_history.csv')))
    flist.sort()

    if flist:
        allperf = pd.concat([
            pd.read_csv(file, index_col=None, header=None,
                        names=[os.path.basename(os.path.split(file)[0])])
            for file in flist
        ], axis=1)
        allperf.to_csv(os.path.join(path, 'allperf_history.csv'), index=False)

    # get HP history for each generation and concatenate into a single file
    flist = list(glob.iglob(os.path.join(path, 'g' + gnames[-1] + '_w*/hp_history.csv')))
    flist.sort()
    for hp in hp_names:
        try:
            # Updated for pandas 2.x: squeeze is deprecated, use squeeze("columns")
            allhp = pd.concat([
                pd.read_csv(file, index_col=None, header=0, usecols=[hp]).squeeze("columns")
                for file in flist
            ], axis=1)
            allhp.to_csv(os.path.join(outpath, hp + '.csv'), index=False)
        except Exception as e:
            print(f"Warning: Could not process hyperparameter {hp}: {e}")

    print(f"Files saved in {path}")


def get_digits(instring):
    """
    Extract digits from a string: abc234de56 --> 23456

    Args:
        instring: Input string

    Returns:
        String containing only digits
    """
    d = ''.join([c for c in instring if c.isdigit()])
    return d


def concat_logfit(path):
    """
    Concatenate fit logs from all generations for each worker.

    Args:
        path: Path to PBT run directory
    """
    outpath = path
    flist = list(glob.iglob(os.path.join(path, '*/fitlog.csv')))
    dnames = [os.path.basename(os.path.split(file)[0]) for file in flist]

    if not dnames:
        return

    # unique generators
    gnames = list(set([get_digits(d.split('_')[0]) for d in dnames]))
    gnames.sort(key=int)
    # unique workers
    wnames = list(set([get_digits(d.split('_')[1]) for d in dnames]))
    wnames.sort(key=int)

    # concatenate the fitlog for each worker
    for w in wnames:
        with open(os.path.join(outpath, 'w' + w + "_fitlog.csv"), 'w') as fout:
            for g in gnames:
                file_list = list(glob.iglob(os.path.join(path, 'g' + g + '_w' + w + '/fitlog.csv')))
                if not file_list:
                    break
                else:
                    file = file_list[0]
                with open(file, 'r') as logfile:
                    fout.write(logfile.read())


if __name__ == "__main__":
    CLI = argparse.ArgumentParser(description='Gather and process PBT run results')
    CLI.add_argument(
        "--topdir",
        nargs="*",
        type=str,
        default='.',
        help='Top directory containing PBT run results'
    )
    CLI.add_argument(
        "--hplist",
        nargs="*",
        type=str,
        default=[],
        help='List of hyperparameter names to extract'
    )

    # parse the command line
    args = CLI.parse_args()

    topdir = args.topdir[0] if isinstance(args.topdir, list) else args.topdir
    gather_run_csv(topdir, args.hplist)
    concat_logfit(topdir)
