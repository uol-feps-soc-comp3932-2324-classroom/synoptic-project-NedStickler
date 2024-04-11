
import keras
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
import numpy as np
from models import SRGAN, SRResNet
from losses import Losses
from utils import GANSaver
from loaders import load_resisc45_subset
import paths


class Training():
    def __init__(self, model: str, epochs: int, first_pass: bool) -> None:
        self.model = model
        self.losses = Losses()
        self.epochs = epochs
        self.first_pass = first_pass
        self.train_data, self.train_labels = load_resisc45_subset("train")
        self.val_data, self.val_labels = load_resisc45_subset("val")

    def _get_model_paths(self, loss: str) -> tuple[str, str]:
        discriminator_path = paths.REPO_PATH + f"/generators/srgan-{loss}/srgan-{loss}-e159-lr0.0001-resics45/discriminator.keras"
        generator_path = paths.REPO_PATH + f"/generators/srgan-{loss}/srgan-{loss}-e159-lr0.0001-resics45/generator.keras"
        return discriminator_path, generator_path
    
    def train_srresnet_mse(self) -> None:
        save_checkpoint = ModelCheckpoint(paths.SAVE_PATH + f"/srresnet-mse/srresnet-mse-e{self.epochs}-resics45.keras", save_best_only=True, mode="auto", save_freq="epoch")
        srresnet = SRResNet(residual_blocks=16, downsample_factor=4)
        srresnet.compile(optimiser=keras.optimizers.Adam(learning_rate=10**-4), loss=keras.losses.MeanSquaredError())
        srresnet.fit(self.train_data, batch_size=15, epochs=self.epochs, validation_data=self.val_data, callbacks=[save_checkpoint])
    
    def train_srgan(self, perceptual_loss: int, discriminator_path: str = None, generator_path: str = None) -> None:
        if self.first_pass:
            discriminator = None
            generator = keras.saving.load_model(paths.REPO_PATH + "/generators/srresnet-mse/srresnet-mse-e1588-resics45.keras")
            lr = 10**-4
        else:
            discriminator = keras.saving.load_model(discriminator_path)
            generator = keras.saving.load_model(generator_path)
            lr = 10**-5
        
        gan_saver = GANSaver(paths.SAVE_PATH, self.model, self.epochs, lr)
        srgan = SRGAN(generator=generator, perceptual_loss=perceptual_loss, discriminator=discriminator)
        srgan.compile(d_optimiser=keras.optimizers.Adam(learning_rate=lr), g_optimiser=keras.optimizers.Adam(learning_rate=lr))
        srgan.fit(self.train_data, batch_size=15, epochs=self.epochs, validation_data=self.val_data, callbacks=[gan_saver])
    
    def train(self) -> None:
        if self.model == "srresnet-mse":
            self.train_srresnet_mse()
        elif self.model == "srgan-vgg22":
            discriminator_path, generator_path = self._get_model_paths("vgg22")
            vgg = self.lossses.vgg19(22)
            self.train_srgan(perceptual_loss=vgg, discriminator_path=discriminator_path, generator_path=generator_path)
        elif self.model == "srgan-vgg54":
            discriminator_path, generator_path = self._get_model_paths("vgg54")
            vgg = self.lossses.vgg19(54)
            self.train_srgan(perceptual_loss=vgg, discriminator_path=discriminator_path, generator_path=generator_path)
        elif self.model == "srgan-xception":
            discriminator_path, generator_path = self._get_model_paths("xception")
            xception = self.losses.xception()
            self.train_srgan(perceptual_loss=xception, discriminator_path=discriminator_path, generator_path=generator_path)
            

if __name__ == "__main__":
    training = Training(model="srgan-vgg22", epochs=159, first_pass=True)
    training.train()
