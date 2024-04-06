
import keras
from keras.layers import Resizing
import numpy as np
from models import SRGAN, SRResNet
from utils import visualise_generator
from layers import GaussianBlur


if __name__ == "__main__":
    generator_path = r"C:\Users\nedst\Desktop\synoptic-project-NedStickler\generators\srgan-vgg54\srgan-vgg54-e67-lr1e-05-resics45\generator.keras"
    hr_images = np.load(r"C:\Users\nedst\Desktop\synoptic-project-NedStickler\datasets\resisc45_test.npy")
    lr_images = GaussianBlur()(hr_images)
    lr_images = Resizing(256 // 4, 256 // 4, interpolation="bicubic")(lr_images).numpy().astype(np.uint8)

    generator = keras.saving.load_model(generator_path)
    visualise_generator(generator, lr_images[:5], hr_images[:5])