
import keras
import matplotlib.pyplot as plt
import numpy as np
import os
from keras import ops
from keras.layers import RandomCrop, Resizing


class GANSaver(keras.callbacks.Callback):
    def __init__(self, save_path: str, model: str, epochs: int, lr: float) -> None:
        super().__init__()
        self.best_loss = 999_999_999
        self.root_path = f"{save_path}/{model}/{model}-e{epochs}-lr{lr}-resics45/"
        if not os.path.exists(self.root_path):
            os.mkdir(self.root_path)
    
    def on_epoch_end(self, epoch: int, logs: dict = None) -> None:
        if logs.get("generator_loss") < self.best_loss:
            self.best_loss = logs.get("generator_loss")
            self.model.generator.save(self.root_path + "generator.keras")
            self.model.discriminator.save(self.root_path + "discriminator.keras")


def visualise_generator(generator: keras.Model, lr_imgs: np.array, hr_imgs: np.array) -> None:
    sr_imgs = generator(lr_imgs)
    num_sr_imgs = len(sr_imgs)

    fig, axs = plt.subplots(num_sr_imgs, 3)
    fig.set_size_inches(9, num_sr_imgs * 3)
    axs[0, 0].set_title("LR")
    axs[0, 1].set_title("HR")
    axs[0, 2].set_title("SR")

    for i, img in enumerate(sr_imgs):
        axs[i, 0].imshow(lr_imgs[i])
        axs[i, 1].imshow(hr_imgs[i])
        axs[i, 2].imshow(img.numpy().astype(np.uint8))
    plt.show()


def crop_and_resize_image(image: np.array, downsample_factor: int) -> keras.Model:
    hr_patch = RandomCrop(96, 96)(image)
    lr_patch = Resizing(96 // downsample_factor, 96 // downsample_factor, interpolation="bicubic")(hr_patch)
    return lr_patch, hr_patch


def crop_and_resize_batch(batch: np.array, downsample_factor: int) -> np.array:
    lr_images = []
    hr_images = []
    for _ in range(16):
        lr_image, hr_image = crop_and_resize_image(batch, downsample_factor)
        lr_images.append(lr_image)
        hr_images.append(hr_image)
    return ops.concatenate(lr_images), ops.concatenate(hr_images)
