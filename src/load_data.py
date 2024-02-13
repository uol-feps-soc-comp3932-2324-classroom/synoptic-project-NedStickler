
import numpy as np
import tensorflow_datasets as tfds
from pathlib import Path


def load_data(package_path: str | Path) -> np.array:
    """Load the RESICS45 dataset using tensorflow_datasets.
    
    Args:
        package_path: Path to the 'tensorflow_datasets' package (./venv/Lib/site-packages/tensorflow_datasets/).

    Returns:
        A numpy array containing the dataset with shape (31_500, 256, 256, 3).
    """
    dataset = tfds.load("resisc45", data_dir=package_path)
    dataset_as_numpy = tfds.as_numpy(dataset)
    images = np.array([feature.get("image") for feature in dataset_as_numpy.get("train")])
    return images