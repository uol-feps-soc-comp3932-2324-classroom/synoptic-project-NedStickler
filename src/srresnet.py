import keras
from keras import layers
from tensorflow.nn import depth_to_space
import numpy as np
from keras.callbacks import ModelCheckpoint


@keras.saving.register_keras_serializable()
class PixelShuffle(keras.Layer):
    def call(self, x):
        return depth_to_space(x, 2)
    
    def get_config(self):
        return {}
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

def g_residual_block(x_in):
    x = layers.Conv2D(64, kernel_size=3, padding="same")(x_in)
    x = layers.BatchNormalization()(x)
    x = layers.PReLU(shared_axes=[1, 2])(x)
    x = layers.Conv2D(64, kernel_size=3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x_in, x])
    return x

def generator(residual_blocks):
    # LR input
    inputs = layers.Input((None, None, 3))
    x_in = layers.Rescaling(scale=1/255)(inputs)

    # First convolution
    x_in = layers.Conv2D(64, kernel_size=3, padding="same")(x_in)
    x_in = x = layers.PReLU(shared_axes=[1, 2])(x_in)

    # Residual block set
    for _ in range(residual_blocks):
        x = g_residual_block(x)
    
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


if __name__ == "__main__":
    residual_blocks = 16
    downsample_factor = 4

    save_checkpoint = ModelCheckpoint("/tmp/sc20ns/generators/srresnet_1_s2048e300b32/srresnet_s2048e300b32.keras", monitor="loss", save_best_only=True, mode="auto", save_freq=64)

    dataset = np.load("/uolstore/home/users/sc20ns/Documents/synoptic-project-NedStickler/datasets/resics45_s2048.npy")
    lr_dataset = np.array([image[::downsample_factor, ::downsample_factor, :] for image in dataset])
    
    # Pre-train generator (SRResNet)
    pre_trained_generator = generator(residual_blocks)
    pre_trained_generator.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0003), loss=keras.losses.MeanSquaredError())
    pre_trained_generator.fit(lr_dataset, dataset, epochs=300, callbacks=[save_checkpoint])
    
    pre_trained_generator.save(r"/tmp/sc20ns/generators/srresnet_1_s2048e300b32/srresnet_s2048e300b32_final.keras")
