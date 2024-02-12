
import numpy as np
import tensorflow_datasets as tfds
from tensorflow_datasets.core.dataset_utils import _IterableDataset


def load_data(venv_path: str) -> _IterableDataset:
    dataset = tfds.load("resisc45", data_dir=f"{venv_path}\\Lib\\site-packages\\tensorflow_datasets\\")
    dataset_as_numpy = tfds.as_numpy(dataset)
    images = np.array([feature.get("image") for feature in dataset_as_numpy.get("train")])
    dataset_as_numpy = None
    return images