
import numpy as np
import tensorflow_datasets as tfds
import paths
from pathlib import Path
from utils import crop_and_resize


def load_resics45(package_path: str | Path) -> np.array:
    """Load the RESISC45 dataset using tensorflow_datasets.
    
    Args:
        package_path: Path to the 'tensorflow_datasets' package (./venv/Lib/site-packages/tensorflow_datasets/).

    Returns:
        A numpy array containing the dataset with shape (31_500, 256, 256, 3).
    """
    dataset = tfds.load("resisc45", data_dir=package_path)
    dataset_as_numpy = tfds.as_numpy(dataset)
    images = np.array([feature.get("image") for feature in dataset_as_numpy.get("train")])
    labels = np.array([feature.get("label") for feature in dataset_as_numpy.get("train")])
    return images, labels


def split_resisc45()


def load_resics45_subset(size: int = 2048, downsample_factor: int = 4) -> np.array:
    if size > 2048:
        size = 2048

    dataset = np.load(paths.REPO_PATH + "/datasets/resics45_s2048.npy")[:size]
    lr_dataset = []
    hr_dataset = []
    
    for image in dataset:
        for _ in range(16):
            lr_image, hr_image = crop_and_resize(image, downsample_factor)
            lr_dataset.append(lr_image)
            hr_dataset.append(hr_image)
    return np.array(lr_dataset), np.array(hr_dataset)
