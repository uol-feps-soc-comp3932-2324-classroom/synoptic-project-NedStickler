import keras
import matplotlib.pyplot as plt
import numpy as np
import os


class GANSaver(keras.callbacks.Callback):
    def __init__(self, save_path: str, details: dict) -> None:
        super().__init__()
        self.best_loss = 999_999_999
        self.root_path = f"{save_path}/{details.model}/{details.model}-e{details.epochs}-lr{details.lr}-resics45/"
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