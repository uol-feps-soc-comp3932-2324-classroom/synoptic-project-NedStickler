import keras
import numpy as np
from models import SRGAN
from utils import visualise_generator


if __name__ == "__main__":
    downsample_factor = 4
    hr_dataset = np.load(r"C:\Users\nedst\Desktop\synoptic-project-NedStickler\datasets\resics45_s2048.npy")
    lr_dataset = lr_dataset = np.array([image[::downsample_factor, ::downsample_factor, :] for image in hr_dataset])

    generator = keras.saving.load_model(r"C:\Users\nedst\Desktop\synoptic-project-NedStickler\generators\srresnet-mse\srresnet-mse-e1-resics45.keras")
    visualise_generator(generator, lr_dataset[:5], hr_dataset[:5])