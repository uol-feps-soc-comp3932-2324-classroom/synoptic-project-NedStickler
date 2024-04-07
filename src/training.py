
import keras
from keras.callbacks import ModelCheckpoint
import numpy as np
from models import SRGAN, SRResNet
from utils import GANSaver
from loaders import load_resisc45_subset
import paths

# TODO:
# Implement dataset with train, test, validation sets
# Implement losses file with predecided losses
# Re-train models
# Need to mention why only 8 patches are taken
# https://keras.io/api/applications/ list of pre-trained models


class Training():
    def __init__(self, model: str, epochs: int) -> None:
        self.model = model
        self.epochs = epochs
        self.vgg_base = keras.applications.VGG19(input_shape=(None, None, 3), weights="imagenet", include_top=False)
        self.train, self.train_labels = load_resisc45_subset("train")
        self.val, self.val_labels = load_resisc45_subset("val")

    def _get_model_paths(self, vgg: str) -> tuple[str, str]:
        discriminator_path = paths.REPO_PATH + f"/generators/srgan-{vgg}/srgan-{vgg}-e67-lr0.0001-resics45/discriminator.keras"
        generator_path = paths.REPO_PATH + f"/generators/srgan-{vgg}/srgan-{vgg}-e67-lr0.0001-resics45/generator.keras"
        return discriminator_path, generator_path
    
    def train_srresnet_mse(self) -> None:
        save_checkpoint = ModelCheckpoint(paths.SAVE_PATH + f"/srresnet-mse/srresnet-mse-e{self.epochs}-resics45.keras", monitor="loss", save_best_only=True, mode="auto", save_freq="epoch")
        srresnet = SRResNet(residual_blocks=16, downsample_factor=4)
        srresnet.compile(optimiser=keras.optimizers.Adam(learning_rate=10**-4), loss=keras.losses.MeanSquaredError())
        srresnet.fit(self.train, batch_size=15, epochs=self.epochs, validation_data=self.val, callbacks=[save_checkpoint])
    
    def train_srgan(self, first_pass: bool, vgg: int, discriminator_path: str = None, generator_path: str = None) -> None:
        if first_pass:
            discriminator = None
            generator = keras.saving.load_model(paths.REPO_PATH + "/generators/srresnet-mse/srresnet-mse-e667-resics45-patch.keras")
            lr = 10**-4
        else:
            discriminator = keras.saving.load_model(discriminator_path)
            generator = keras.saving.load_model(generator_path)
            lr = 10**-5

        gan_saver = GANSaver(paths.SAVE_PATH, self.model, self.epochs, lr)
        srgan = SRGAN(generator=generator, vgg=vgg, discriminator=discriminator)
        srgan.compile(d_optimiser=keras.optimizers.Adam(learning_rate=lr), g_optimiser=keras.optimizers.Adam(learning_rate=lr))
        srgan.fit(self.train, batch_size=30, epochs=self.epochs, callbacks=[gan_saver])
    
    def train_vgg(self) -> None:
        save_checkpoint = ModelCheckpoint(paths.SAVE_PATH + f"/vgg/vgg-e{self.epochs}-resics45.weights.h5", monitor="loss", save_weights_only=True, save_best_only=True, mode="auto", save_freq="epoch")
        vgg = keras.applications.VGG19(input_shape=(256, 256, 3), weights=None, classes=45)
        vgg.compile(optimizer="adam", loss="categorical_crossentropy")
        vgg.fit(keras.applications.vgg19.preprocess_input(self.train), keras.utils.to_categorical(self.labels), epochs=self.epochs, callbacks=[save_checkpoint])
    
    def train(self) -> None:
        if self.model == "srresnet-mse":
            self.train_srresnet_mse()
        elif self.model == "srgan-vgg22":
            discriminator_path, generator_path = self._get_model_paths("vgg22")
            vgg = keras.Model(self.vgg_base.input, self.vgg_base.layers[5].output)  
            self.train_srgan(first_pass=False, vgg=vgg, discriminator_path=discriminator_path, generator_path=generator_path)
        elif self.model == "srgan-vgg54":
            discriminator_path, generator_path = self._get_model_paths("vgg54")
            vgg = keras.Model(self.vgg_base.input, self.vgg_base.layers[20].output)  
            self.train_srgan(first_pass=True, vgg=vgg, discriminator_path=discriminator_path, generator_path=generator_path)
        elif self.model == "vgg":
            self.train_vgg() 


if __name__ == "__main__":
    training = Training(model="srresnet-mse", epochs=5)
    training.train()
