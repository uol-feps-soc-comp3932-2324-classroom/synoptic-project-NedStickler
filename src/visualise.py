
import keras
import numpy as np
from models import SRGAN, SRResNet
from utils import visualise_generator
from layers import PixelShuffle


if __name__ == "__main__":
    generator_path = r"C:\Users\nedst\Desktop\synoptic-project-NedStickler\generators\srresnet-mse\srresnet-mse-e3-resics45.keras"
    downsample_factor = 4
    hr_dataset = np.load(r"C:\Users\nedst\Desktop\synoptic-project-NedStickler\datasets\resics45_s2048.npy")
    lr_dataset = lr_dataset = np.array([image[::downsample_factor, ::downsample_factor, :] for image in hr_dataset])

    generator = keras.saving.load_model(generator_path)
    generator.compile(optimiser=keras.optimizers.Adam(learning_rate=10**-4), loss=keras.losses.MeanSquaredError())
    visualise_generator(generator, lr_dataset[5:10], hr_dataset[5:10])