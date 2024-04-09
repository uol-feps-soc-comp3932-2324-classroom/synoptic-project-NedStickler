
import keras
import matplotlib.pyplot as plt
import numpy as np
import os
from keras import ops
from keras.layers import RandomCrop, Resizing
import json


class GANSaver(keras.callbacks.Callback):
    def __init__(self, save_path: str, model: str, epochs: int, lr: float) -> None:
        super().__init__()
        self.best_loss = 999_999_999
        self.root_path = f"{save_path}/{model}/{model}-e{epochs}-lr{lr}-resics45/"
        if not os.path.exists(self.root_path):
            os.mkdir(self.root_path)
    
    def on_epoch_end(self, epoch: int, logs: dict = None) -> None:
        if logs.get("val_generator_loss") < self.best_loss:
            self.best_loss = logs.get("val_generator_loss")
            self.model.generator.save(self.root_path + "generator.keras")
            self.model.discriminator.save(self.root_path + "discriminator.keras")
            with open(self.root_path + "save_epoch.json", "w") as f:
                f.write(json.dumps({"save_epoch": epoch + 1}))
            np.save(self.model.d_optimiser.get_weights(), self.root_path + "d_optimiser_weights.npy")
            np.save(self.model.g_optimiser.get_weights(), self.root_path + "g_optimiser_weights.npy")


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
