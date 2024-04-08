
import numpy as np
import tensorflow_datasets as tfds
import paths
from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import Literal


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


def train_test_split_resisc45(package_path: str | Path, train_size, val_size, test_size) -> tuple[np.array, np.array]:
    images, labels = load_resisc45(package_path)
    X_train, X_val_and_test, y_train, y_val_and_test = train_test_split(images, labels, train_size=train_size/31500, test_size=(val_size+test_size)/31500, random_state=42, stratify=labels)
    X_val, X_test, y_val, y_test = train_test_split(X_val_and_test, y_val_and_test, train_size=0.5, random_state=42, stratify=y_val_and_test)
    return X_train, X_val, X_test, y_train, y_val, y_test


def generate_resisc45_files(save_path: str | Path, package_path: str | Path, train_size: int, val_size: int, test_size: int) -> None:
    X_train, X_val, X_test, y_train, y_val, y_test = train_test_split_resisc45(package_path, train_size, val_size, test_size)
    np.save(save_path + r"\resisc45_train.npy", X_train)
    np.save(save_path + r"\resisc45_val.npy", X_val)
    np.save(save_path + r"\resisc45_test.npy", X_test)
    np.save(save_path + r"\resisc45_train_labels.npy", y_train)
    np.save(save_path + r"\resisc45_val_labels.npy", y_val)
    np.save(save_path + r"\resisc45_test_labels.npy", y_test)


def load_resisc45_subset(dataset_type: Literal["train", "val", "test"]) -> np.array:
    return np.load(paths.REPO_PATH + f"/datasets/resisc45_{dataset_type}.npy"), np.load(paths.REPO_PATH + f"/datasets/resisc45_{dataset_type}_labels.npy")


if __name__ == "__main__":
    package_path = r"C:\Users\nedst\Desktop\synoptic-project-NedStickler\.venv\Lib\site-packages\tensorflow_datasets"
    images, labels = load_resisc45(package_path)
    generate_resisc45_files(r"C:\Users\nedst\Desktop\synoptic-project-NedStickler\datasets", package_path, 945, 135, 135)


def get_label_mapping() -> dict:
    return {
        0: 'airplane',
        1: 'airport',
        2: 'baseball_diamond',
        3: 'basketball_court',
        4: 'beach',
        5: 'bridge',
        6: 'chaparral',
        7: 'church',
        8: 'circular_farmland',
        9: 'cloud',
        10: 'commercial_area',
        11: 'dense_residential',
        12: 'desert',
        13: 'forest',
        14: 'freeway',
        15: 'golf_course',
        16: 'ground_track_field',
        17: 'harbor',
        18: 'industrial_area',
        19: 'intersection',
        20: 'island',
        21: 'lake',
        22: 'meadow',
        23: 'medium_residential',
        24: 'mobile_home_park',
        25: 'mountain',
        26: 'overpass',
        27: 'palace',
        28: 'parking_lot',
        29: 'railway',
        30: 'railway_station',
        31: 'rectangular_farmland',
        32: 'river',
        33: 'roundabout',
        34: 'runway',
        35: 'sea_ice',
        36: 'ship',
        37: 'snowberg',
        38: 'sparse_residential',
        39: 'stadium',
        40: 'storage_tank',
        41: 'tennis_court',
        42: 'terrace',
        43: 'thermal_power_station',
        44: 'wetland'
    }
