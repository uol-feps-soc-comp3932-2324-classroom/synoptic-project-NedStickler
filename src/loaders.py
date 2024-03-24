
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


def load_resics45_subset() -> np.array:
    dataset = np.load(paths.REPO_PATH + "/datasets/resics45_s2048.npy")
    lr_dataset = []
    hr_dataset = []
    
    for image in dataset:
        lr_image, hr_image = crop_and_resize(image, 4)
        lr_dataset.append(lr_image)
        hr_dataset.append(hr_image)
    return np.array(lr_dataset), np.array(hr_dataset)
