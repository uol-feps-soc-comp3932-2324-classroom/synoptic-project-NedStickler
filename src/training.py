
import keras
from keras.callbacks import ModelCheckpoint
import numpy as np
from models import SRGAN, SRResNet
from utils import GANSaver
from loaders import load_resics45_subset
import paths


class Training():
    def __init__(self, model: str, epochs: int) -> None:
        self.model = model
        self.epochs = epochs
        self.lr_dataset, self.hr_dataset = load_resics45_subset()
    
    def train_srresnet_mse(self) -> None:
        save_checkpoint = ModelCheckpoint(paths.SAVE_PATH + f"/srresnet-mse/srresnet-mse-e{self.epochs}-resics45.keras", monitor="loss", save_best_only=True, mode="auto", save_freq="epoch")
        srresnet = SRResNet(16)
        srresnet.compile(optimizer=keras.optimizers.Adam(learning_rate=10**-4), loss=keras.losses.MeanSquaredError())
        srresnet.fit(self.lr_dataset, self.hr_dataset, epochs=self.epochs, callbacks=[save_checkpoint])
    
    def train_srgan(self, first_pass: bool, vgg: int, discriminator_path: str = None, generator_path: str = None) -> None:
        if first_pass:
            discriminator = None
            generator = keras.saving.load_model(paths.REPO_PATH + "/generators/srresnet-mse/srresnet-mse-e1000-resics45.keras")
            lr = 10**-4
        else:
            discriminator = keras.saving.load_model(discriminator_path)
            generator = keras.saving.load_model(generator_path)
            lr = 10**-5

        if vgg == 22:
            vgg_layer = 5
        elif vgg == 54:
            vgg_layer = 20

        gan_saver = GANSaver(paths.SAVE_PATH, self.model, self.epochs, lr)
        vgg = keras.applications.VGG19(input_shape=(None, None, 3), weights="imagenet", include_top=False)
        vgg = keras.Model(vgg.input, vgg.layers[vgg_layer].output)

        srgan = SRGAN(generator=generator, vgg=vgg, discriminator=discriminator)
        srgan.compile(d_optimiser=keras.optimizers.Adam(learning_rate=lr), g_optimiser=keras.optimizers.Adam(learning_rate=lr))
        srgan.fit(self.lr_dataset, self.hr_dataset, epochs=self.epochs, callbacks=[gan_saver])
    
    def train(self) -> None:
        if self.model == "srresnet-mse":
            self.train_srresnet_mse()
        elif self.model == "srgan-vgg22":
            discriminator_path = paths.REPO_PATH + "/generators/srgan-vgg22/srgan-vgg22-e100-lr0.0001-resics45/discriminator.keras"
            generator_path = paths.REPO_PATH + "/generators/srgan-vgg22/srgan-vgg22-e100-lr0.0001-resics45/generator.keras"
            self.train_srgan(first_pass=True, vgg=22, discriminator_path=discriminator_path, generator_path=generator_path)
        elif self.model == "srgan-vgg54":
            discriminator_path = paths.REPO_PATH + "/generators/srgan-vgg54/srgan-vgg54-e100-lr0.0001-resics45/discriminator.keras"
            generator_path = paths.REPO_PATH + "/generators/srgan-vgg54/srgan-vgg54-e100-lr0.0001-resics45/generator.keras"
            self.train_srgan(first_pass=True, vgg=54, discriminator_path=discriminator_path, generator_path=generator_path)


if __name__ == "__main__":
    training = Training(model="srresnet-mse", epochs=7)
    training.train()
