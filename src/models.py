
import tensorflow as tf
import keras
from keras import layers
import numpy as np
from layers import PixelShuffle
from utils import crop_and_resize_batch


# def SRResNet(residual_blocks: int) -> keras.Model:
#     def g_residual_block(x_in):
#         x = layers.Conv2D(64, kernel_size=3, padding="same")(x_in)
#         x = layers.BatchNormalization()(x)
#         x = layers.PReLU(shared_axes=[1, 2])(x)
#         x = layers.Conv2D(64, kernel_size=3, padding="same")(x)
#         x = layers.BatchNormalization()(x)
#         x = layers.Add()([x_in, x])
#         return x
    
#     # LR input
#     inputs = layers.Input((None, None, 3))
#     x_in = layers.Rescaling(scale=1/255)(inputs)
#     # First convolution
#     x_in = layers.Conv2D(64, kernel_size=3, padding="same")(x_in)
#     x_in = x = layers.PReLU(shared_axes=[1, 2])(x_in)

#     # Residual block set
#     for _ in range(residual_blocks):
#         x = g_residual_block(x)

#     # Residual block without activation functions
#     x = layers.Conv2D(64, kernel_size=3, padding="same")(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.Add()([x_in, x])
#     # Upscaling blocks
#     x = layers.Conv2D(256, kernel_size=3, padding="same")(x)
#     x = PixelShuffle()(x)
#     x = layers.PReLU(shared_axes=[1, 2])(x)
#     x = layers.Conv2D(256, kernel_size=3, padding="same")(x)
#     x = PixelShuffle()(x)
#     x = layers.PReLU(shared_axes=[1, 2])(x)
#     # Final convolve
#     x = layers.Conv2D(3, kernel_size=3, padding="same")(x)
#     x = layers.Rescaling(scale=127.5, offset=127.5)(x)
#     return keras.Model(inputs, x)


@keras.saving.register_keras_serializable()
class SRResNet(keras.Model):
    def __init__(self, residual_blocks: int, downsample_factor: int) -> None:
        super().__init__()
        self.residual_blocks = residual_blocks
        self.downsample_factor = downsample_factor
        self.model = self.get_model()
    
    def get_config(self) -> dict:
        return {"model": self.model}
    
    @classmethod
    def from_config(cls, config) -> dict:
        return cls(**config)
    
    def _residual_block(self, x_in):
        x = layers.Conv2D(64, kernel_size=3, padding="same")(x_in)
        x = layers.BatchNormalization()(x)
        x = layers.PReLU(shared_axes=[1, 2])(x)
        x = layers.Conv2D(64, kernel_size=3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Add()([x_in, x])
        return x

    def get_model(self):
        # LR input
        inputs = layers.Input((None, None, 3))
        x_in = layers.Rescaling(scale=1/255)(inputs)
        # First convolution
        x_in = layers.Conv2D(64, kernel_size=3, padding="same")(x_in)
        x_in = x = layers.PReLU(shared_axes=[1, 2])(x_in)

        # Residual block set
        for _ in range(self.residual_blocks):
            x = self._residual_block(x)

        # Residual block without activation functions
        x = layers.Conv2D(64, kernel_size=3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Add()([x_in, x])
        # Upscaling blocks
        x = layers.Conv2D(256, kernel_size=3, padding="same")(x)
        x = PixelShuffle()(x)
        x = layers.PReLU(shared_axes=[1, 2])(x)
        x = layers.Conv2D(256, kernel_size=3, padding="same")(x)
        x = PixelShuffle()(x)
        x = layers.PReLU(shared_axes=[1, 2])(x)
        # Final convolve
        x = layers.Conv2D(3, kernel_size=3, padding="same")(x)
        x = layers.Rescaling(scale=127.5, offset=127.5)(x)
        return keras.Model(inputs, x)
    
    def compile(self, optimiser, loss):
        super().compile()
        self.optimiser = optimiser
        self.loss = loss

    def train_step(self, data):
        lr_images, hr_images = crop_and_resize_batch(data, self.downsample_factor)
        with tf.GradientTape() as tape:
            sr_images = self.model(lr_images)
            loss = self.loss(hr_images, sr_images)
        gradients = tape.gradient(loss, self.model.trainable_weights)
        self.optimiser.apply(gradients(zip(gradients, self.model.trainable_weights)))
        return {"loss": loss}


@keras.saving.register_keras_serializable()
class SRGAN(keras.Model):
    def __init__(self, generator: keras.Model, vgg: keras.Model, discriminator: keras.Model = None):
        super().__init__()
        if discriminator is None: self.discriminator = self.get_discriminator()
        else: self.discriminator = discriminator
        self.generator = generator
        self.vgg = vgg
        self.bce_loss = keras.losses.BinaryCrossentropy()
        self.mse_loss = keras.losses.MeanSquaredError()
        self.g_loss_tracker = keras.metrics.Mean(name="generator_loss")
    
    def _d_residual_block(self, x, n_filters: int, n_strides: int):
        x = layers.Conv2D(n_filters, kernel_size=3, strides=n_strides, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(0.2)(x)
        return x

    def _d_downsample_pair(self, x, n_filters: int):
        x = self._d_residual_block(x, n_filters, 1)
        x = self._d_residual_block(x, n_filters, 2)
        return x
    
    def get_discriminator(self) -> keras.Model:
        # HR/SR input
        inputs = layers.Input((None, None, 3))
        x = layers.Rescaling(scale=1/127.5, offset=-1)(inputs)
        # First convolution blocks
        x = layers.Conv2D(64, kernel_size=3, padding="same")(x)
        x = layers.LeakyReLU(0.2)(x)
        # Residual downsampling blocks
        x = self._d_residual_block(x, 64, 2)
        x = self._d_downsample_pair(x, 128)
        x = self._d_downsample_pair(x, 256)
        x = self._d_downsample_pair(x, 512)
        # Flatten and classify
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.LeakyReLU(0.2)(x)
        x = layers.Dense(1, activation="sigmoid")(x)

        return keras.Model(inputs, x)
    
    def get_config(self) -> dict:
        return {
            "generator": self.generator,
            "discriminator": self.discriminator,
            "vgg": self.vgg,
        }
    
    @classmethod
    def from_config(cls, config) -> dict:
        return cls(**config)
    
    @property
    def metrics(self) -> list:
        return [self.g_loss_tracker]

    def compile(self, d_optimiser, g_optimiser) -> None:
        super().compile()
        self.d_optimiser = d_optimiser
        self.g_optimiser = g_optimiser

    def train_step(self, data: np.array) -> dict:
        lr_images, hr_images = data
        batch_size = lr_images.shape[0]

        # Train the discriminator
        generated_images = self.generator(lr_images)
        combined_images = keras.ops.concatenate([generated_images, hr_images])
        d_labels = np.concatenate([np.ones((batch_size, 1)), np.zeros((batch_size, 1))])

        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_images)
            d_loss = self.bce_loss(d_labels, predictions)
        gradients = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimiser.apply_gradients(zip(gradients, self.discriminator.trainable_weights))

        # Train the generator
        misleading_labels = np.zeros((batch_size, 1))

        with tf.GradientTape() as tape:
            generated_images = self.generator(lr_images)
            predictions = self.discriminator(generated_images)
            g_loss = 0.001 * self.bce_loss(misleading_labels, predictions)
            sr_vgg = tf.keras.applications.vgg19.preprocess_input(generated_images)
            sr_vgg = self.vgg(sr_vgg) / 12.75
            hr_vgg = tf.keras.applications.vgg19.preprocess_input(hr_images)
            hr_vgg = self.vgg(hr_vgg) / 12.75
            perceptual_loss = self.mse_loss(hr_vgg, sr_vgg)
            g_total_loss = g_loss + perceptual_loss
        
        gradients = tape.gradient(g_total_loss, self.generator.trainable_weights)
        self.g_optimiser.apply_gradients(zip(gradients,self.generator.trainable_weights))
        self.g_loss_tracker.update_state(g_total_loss)

        losses = {
            "loss": g_total_loss,
            "d_loss": d_loss,
            "g_total_loss": g_total_loss,
            "g_loss": g_loss,
            "pereceptual_loss": perceptual_loss
        }

        return losses
