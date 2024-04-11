
import keras
from keras.layers import Resizing
import numpy as np
from models import SRGAN, SRResNet, BlurAndResize
from utils import visualise_generator


if __name__ == "__main__":
    generator_path = r"C:\Users\nedst\Desktop\synoptic-project-NedStickler\generators\srgan-vgg54\srgan-vgg54-e159-lr1e-05-resics45\generator.keras"
    hr_images = np.load(r"C:\Users\nedst\Desktop\synoptic-project-NedStickler\datasets\resisc45_test.npy")
    lr_images = BlurAndResize(4)(hr_images).numpy().astype(np.uint8)
    generator = keras.saving.load_model(generator_path)
    visualise_generator(generator, lr_images[:3], hr_images[:3])