#!/usr/bin/env python3
"""
TensorFlow 2.x Compatibility Module for AutoLFADS/RADICaL
Provides abstraction layer for file I/O operations that works with both local and GCS paths.
"""

import os
import sys

# Try to import TensorFlow 2.x gfile utilities
try:
    import tensorflow as tf
    # TensorFlow 2.x uses tf.io.gfile
    if hasattr(tf.io, 'gfile'):
        _USE_TF_GFILE = True
    else:
        _USE_TF_GFILE = False
except ImportError:
    _USE_TF_GFILE = False
    tf = None

# Fallback to google-cloud-storage for GCS paths
try:
    from google.cloud import storage
    _HAS_GCS = True
except ImportError:
    _HAS_GCS = False


def _is_gcs_path(path):
    """Check if path is a Google Cloud Storage path."""
    if path is None:
        return False
    return str(path).startswith('gs://')


def file_exists(path):
    """
    Check if a file exists.
    Works with local paths and GCS paths (gs://).
    """
    if path is None or path == -1:
        return False

    path = str(path)

    if _is_gcs_path(path):
        if _USE_TF_GFILE:
            return tf.io.gfile.exists(path)
        elif _HAS_GCS:
            # Parse gs://bucket/path format
            parts = path[5:].split('/', 1)
            bucket_name = parts[0]
            blob_name = parts[1] if len(parts) > 1 else ''
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            return blob.exists()
        else:
            raise ImportError("Neither TensorFlow nor google-cloud-storage is available for GCS operations")
    else:
        return os.path.exists(path)


def makedirs(path, exist_ok=True):
    """
    Create directory and any necessary parent directories.
    Works with local paths and GCS paths.
    """
    path = str(path)

    if _is_gcs_path(path):
        if _USE_TF_GFILE:
            tf.io.gfile.makedirs(path)
        # GCS doesn't need explicit directory creation
        return
    else:
        os.makedirs(path, exist_ok=exist_ok)


def file_open(path, mode='r'):
    """
    Open a file for reading or writing.
    Works with local paths and GCS paths.

    Returns a file-like object that can be used as a context manager.
    """
    path = str(path)

    if _is_gcs_path(path):
        if _USE_TF_GFILE:
            return tf.io.gfile.GFile(path, mode)
        else:
            raise ImportError("TensorFlow is required for GCS file operations")
    else:
        # Ensure directory exists for write modes
        if 'w' in mode or 'a' in mode:
            dir_path = os.path.dirname(path)
            if dir_path and not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)
        return open(path, mode)


def glob(pattern):
    """
    Return list of paths matching pattern.
    Works with local paths and GCS paths.
    """
    pattern = str(pattern)

    if _is_gcs_path(pattern):
        if _USE_TF_GFILE:
            return tf.io.gfile.glob(pattern)
        else:
            raise ImportError("TensorFlow is required for GCS glob operations")
    else:
        import glob as _glob
        return _glob.glob(pattern)


def rmtree(path):
    """
    Recursively delete a directory tree.
    Works with local paths and GCS paths.
    """
    path = str(path)

    if _is_gcs_path(path):
        if _USE_TF_GFILE:
            tf.io.gfile.rmtree(path)
        elif _HAS_GCS:
            # Parse gs://bucket/path format
            parts = path[5:].split('/', 1)
            bucket_name = parts[0]
            prefix = parts[1] if len(parts) > 1 else ''
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            blobs = bucket.list_blobs(prefix=prefix)
            for blob in blobs:
                blob.delete()
        else:
            raise ImportError("Neither TensorFlow nor google-cloud-storage is available for GCS operations")
    else:
        import shutil
        if os.path.exists(path):
            shutil.rmtree(path)


def copy(src, dst):
    """
    Copy a file from src to dst.
    Works with local paths and GCS paths.
    """
    src = str(src)
    dst = str(dst)

    if _is_gcs_path(src) or _is_gcs_path(dst):
        if _USE_TF_GFILE:
            tf.io.gfile.copy(src, dst, overwrite=True)
        else:
            raise ImportError("TensorFlow is required for GCS copy operations")
    else:
        import shutil
        shutil.copy(src, dst)


def listdir(path):
    """
    List directory contents.
    Works with local paths and GCS paths.
    """
    path = str(path)

    if _is_gcs_path(path):
        if _USE_TF_GFILE:
            return tf.io.gfile.listdir(path)
        elif _HAS_GCS:
            parts = path[5:].split('/', 1)
            bucket_name = parts[0]
            prefix = parts[1] if len(parts) > 1 else ''
            if prefix and not prefix.endswith('/'):
                prefix += '/'
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            blobs = bucket.list_blobs(prefix=prefix, delimiter='/')
            # Get both blobs and prefixes
            items = []
            for blob in blobs:
                name = blob.name[len(prefix):] if prefix else blob.name
                if name:
                    items.append(name)
            return items
        else:
            raise ImportError("Neither TensorFlow nor google-cloud-storage is available for GCS operations")
    else:
        return os.listdir(path)


def isdir(path):
    """
    Check if path is a directory.
    Works with local paths and GCS paths.
    """
    path = str(path)

    if _is_gcs_path(path):
        if _USE_TF_GFILE:
            return tf.io.gfile.isdir(path)
        else:
            # GCS doesn't have real directories, check if there are files with this prefix
            return True
    else:
        return os.path.isdir(path)


# Convenience aliases
FileIO = file_open
get_matching_files = glob
delete_recursively = rmtree
create_dir = makedirs
