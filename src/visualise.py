
import keras
from keras.layers import Resizing
import numpy as np
from models import SRGAN, SRResNet
from utils import visualise_generator
from loaders import load_resisc45


if __name__ == "__main__":
    generator_path = r"C:\Users\nedst\Desktop\synoptic-project-NedStickler\generators\srresnet-mse\srresnet-mse-e667-resics45-patch.keras"
    hr_images = np.load(r"C:\Users\nedst\Desktop\synoptic-project-NedStickler\datasets\resisc45_test.npy")
    lr_images = Resizing(256 // 4, 256 // 4, interpolation="bicubic")(hr_images).numpy().astype(np.uint8)

    generator = keras.saving.load_model(generator_path)
    generator.compile(optimiser=keras.optimizers.Adam(learning_rate=10**-4), loss=keras.losses.MeanSquaredError())
    visualise_generator(generator, lr_images[5:10], hr_images[5:10])