import keras
from keras.callbacks import ModelCheckpoint
import numpy as np
from models import SRGAN
from utils import GANSaver


def train_srresnet(epochs, save_path):
    save_checkpoint = ModelCheckpoint(save_path, monitor="loss", save_best_only=True, mode="auto", save_freq="epoch")
    vgg = keras.applications.VGG19(input_shape=(None, None, 3), weights="imagenet", include_top=False)
    vgg = keras.Model(vgg.input, vgg.layers[20].output)
    
    srresnet = SRGAN(residual_blocks=16, vgg=vgg).generator
    srresnet.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0003), loss=keras.losses.MeanSquaredError())
    srresnet.fit(lr_dataset, hr_dataset, epochs=epochs, callbacks=[save_checkpoint])


if __name__ == "__main__":
    model = "srresnet-mse"

    epochs = 100 
    save_path = f"/tmp/sc20ns/generators/{model}/{model}-e{epochs}-resics45.keras"
    downsample_factor = 4

    hr_dataset = np.load("/uolstore/home/users/sc20ns/Documents/synoptic-project-NedStickler/datasets/resics45_s2048.npy")
    lr_dataset = np.array([image[::downsample_factor, ::downsample_factor, :] for image in hr_dataset])

    if model.lower() == "srresnet-mse":
        train_srresnet(epochs, save_path)
