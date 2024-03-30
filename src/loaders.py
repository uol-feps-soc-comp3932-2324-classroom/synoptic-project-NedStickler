
import numpy as np
import tensorflow_datasets as tfds
import paths
from pathlib import Path
from sklearn.model_selection import train_test_split


def load_resisc45(package_path: str | Path) -> tuple[np.array, np.array]:
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


def train_test_split_resisc45(package_path: str | Path) -> tuple[np.array, np.array]:
    images, labels = load_resisc45(package_path)
    X_train, X_test = train_test_split(images, train_size=2250/31500, test_size=225/31500, random_state=42, stratify=labels)
    return X_train, X_test


def generate_resisc45_files(path: str | Path) -> None:
    train, test = train_test_split_resisc45(r"C:\Users\nedst\Desktop\synoptic-project-NedStickler\.venv\Lib\site-packages\tensorflow_datasets")
    np.save(path + r"\resisc45_train.npy", train)
    np.save(path + r"\resisc45_test.npy", test)


def load_resisc45(train: bool = True) -> np.array:
    if train: suffix = "train"
    else: suffix = "test"
    return np.load(paths.REPO_PATH + f"/datasets/resisc45_{suffix}.npy")
