import numpy as np
import tensorflow_datasets as tfds
import os
from pathlib import Path


def load_data(venv_name: str) -> np.array:
    dataset = tfds.load("resisc45", data_dir=f".\\{venv_name}\\Lib\\site-packages\\tensorflow_datasets\\")
    dataset_as_numpy = tfds.as_numpy(dataset)
    return np.array([img.get("image") for img in dataset_as_numpy.get("train")])
