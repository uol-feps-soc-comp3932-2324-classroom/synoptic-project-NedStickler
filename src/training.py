import keras
from keras.callbacks import ModelCheckpoint
import numpy as np
from models import SRGAN, SRResNet
from utils import GANSaver


def train_srresnet():
    save_checkpoint = ModelCheckpoint(save_path, monitor="loss", save_best_only=True, mode="auto", save_freq="epoch")
    srresnet = SRResNet(16)
    srresnet.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss=keras.losses.MeanSquaredError())
    srresnet.fit(lr_dataset, hr_dataset, epochs=epochs, callbacks=[save_checkpoint])
    

def train_srgan():
    gan_saver = GANSaver(save_path)
    vgg = keras.applications.VGG19(input_shape=(None, None, 3), weights="imagenet", include_top=False)
    vgg = keras.Model(vgg.input, vgg.layers[20].output)
    
    SRResNet(16)
    srresnet = keras.saving.load_model("/tmp/sc20ns/generators/srresnet-mse/srresnet-mse-e1000-resics45.keras")
    srgan = SRGAN(generator=srresnet, vgg=vgg)
    srgan.compile(d_optimiser=keras.optimizers.Adam(learning_rate=lr), g_optimiser=keras.optimizers.Adam(learning_rate=lr))
    srgan.fit(lr_dataset, hr_dataset, epochs=epochs, callbacks=[gan_saver])


if __name__ == "__main__":
    model = "srgan"
    lr = 10e-4
    epochs = 100
    downsample_factor = 4
    save_path = f"/tmp/sc20ns/generators/{model}/{model}-e{epochs}-{str(lr)}-resics45.keras"

    hr_dataset = np.load("/uolstore/home/users/sc20ns/Documents/synoptic-project-NedStickler/datasets/resics45_s2048.npy")
    lr_dataset = np.array([image[::downsample_factor, ::downsample_factor, :] for image in hr_dataset])

    if model.lower() == "srresnet-mse":
        train_srresnet()
    if model.lower() == "srgan":
        train_srgan()
