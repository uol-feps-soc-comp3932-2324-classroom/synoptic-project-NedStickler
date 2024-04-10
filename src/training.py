
import keras
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
import numpy as np
from models import SRGAN, SRResNet
from utils import GANSaver
from loaders import load_resisc45_subset
import paths
import json


class Training():
    def __init__(self, model: str, epochs: int) -> None:
        self.model = model
        self.epochs = epochs
        self.vgg_base = keras.applications.VGG19(input_shape=(None, None, 3), weights="imagenet", include_top=False)
        self.train_data, self.train_labels = load_resisc45_subset("train")
        self.val_data, self.val_labels = load_resisc45_subset("val")

    def _get_model_paths(self, vgg: str) -> tuple[str, str]:
        discriminator_path = paths.REPO_PATH + f"/generators/srgan-{vgg}/srgan-{vgg}-e159-lr0.0001-resics45/discriminator.keras"
        generator_path = paths.REPO_PATH + f"/generators/srgan-{vgg}/srgan-{vgg}-e159-lr0.0001-resics45/generator.keras"
        return discriminator_path, generator_path
    
    def train_srresnet_mse(self) -> None:
        save_checkpoint = ModelCheckpoint(paths.SAVE_PATH + f"/srresnet-mse/srresnet-mse-e{self.epochs}-resics45.keras", save_best_only=True, mode="auto", save_freq="epoch")
        srresnet = SRResNet(residual_blocks=16, downsample_factor=4)
        srresnet.compile(optimiser=keras.optimizers.Adam(learning_rate=10**-4), loss=keras.losses.MeanSquaredError())
        srresnet.fit(self.train_data, batch_size=15, epochs=self.epochs, validation_data=self.val_data, callbacks=[save_checkpoint])
    
    def train_srgan(self, first_pass: bool, vgg: int, discriminator_path: str = None, generator_path: str = None) -> None:
        if first_pass:
            discriminator = None
            generator = keras.saving.load_model(paths.REPO_PATH + "/generators/srresnet-mse/srresnet-mse-e1588-resics45.keras")
            lr = 10**-4
        else:
            discriminator = keras.saving.load_model(discriminator_path)
            generator = keras.saving.load_model(generator_path)
            lr = 10**-5
        
        gan_saver = GANSaver(paths.SAVE_PATH, self.model, self.epochs, lr)
        srgan = SRGAN(generator=generator, vgg=vgg, discriminator=discriminator)
        srgan.compile(d_optimiser=keras.optimizers.Adam(learning_rate=lr), g_optimiser=keras.optimizers.Adam(learning_rate=lr))
        srgan.fit(self.train_data, batch_size=15, epochs=self.epochs, validation_data=self.val_data, callbacks=[gan_saver])
    
    def resume_sr_gan(self, root_path, vgg, lr):
        with open(root_path + "/save_epoch.json") as f:
            data = json.load(f)
        epochs = 159 - data.get("save_epoch")

        d_optimiser = keras.optimizers.Adam(10**-4)
        discriminator = keras.saving.load_model(root_path + "/discriminator.keras")
        d_optimiser_weights = np.load(root_path + "/d_optimiser_weights.npy")
        d_grad_vars = discriminator.trainable_weights
        d_zero_grads = [tf.zeros_like(w) for w in d_grad_vars]
        d_optimiser.apply_gradients(zip(d_zero_grads, d_grad_vars))
        d_optimiser.set_weights(d_optimiser_weights)

        g_optimiser = keras.optimizers.Adam(10**-4)
        generator = keras.saving.load_model(root_path + "/generator.keras")
        g_optimiser_weights = np.load(root_path + "/g_optimiser_weights.npy")
        g_grad_vars = generator.trainable_weights
        g_zero_grads = [tf.zeros_like(w) for w in g_grad_vars]
        g_optimiser.apply_gradients(zip(g_zero_grads, g_grad_vars))
        g_optimiser.set_weights(g_optimiser_weights)

        gan_saver = GANSaver(paths.SAVE_PATH, "srgan-vgg54", epochs, lr)
        srgan = SRGAN(generator=generator, vgg=vgg, discriminator=discriminator)
        srgan.compile(d_optimiser=d_optimiser, g_optimiser=g_optimiser)
        srgan.fit(self.train_data, batch_size=15, epochs=epochs, validation_data=self.val_data, callbacks=[gan_saver])
    
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
        elif self.model == "resume-srgan-vgg54":
            vgg = keras.Model(self.vgg_base.input, self.vgg_base.layers[20].output)
            root_path = "/tmp/sc20ns/generators/srgan-vgg54/srgan-vgg54-e159-lr0.0001-resics45"
            self.resume_sr_gan(root_path=root_path, vgg=vgg, lr=10**-4)


if __name__ == "__main__":
    training = Training(model="srgan-vgg54", epochs=159)
    training.train()
