import keras
from keras.callbacks import ModelCheckpoint
import numpy as np
from models import SRGAN, SRResNet
from utils import GANSaver
import paths


def train_srresnet_mse():
    save_checkpoint = ModelCheckpoint(paths.SAVE_PATH + "/srresnet-mse/srresnet-mse-e{epochs}-lr{lr}-resics45.keras", monitor="loss", save_best_only=True, mode="auto", save_freq="epoch")
    srresnet = SRResNet(16)
    srresnet.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss=keras.losses.MeanSquaredError())
    srresnet.fit(lr_dataset, hr_dataset, epochs=epochs, callbacks=[save_checkpoint])
    

def train_srgan(first_pass: bool = True):
    gan_saver = GANSaver(paths.SAVE_PATH, details)
    vgg = keras.applications.VGG19(input_shape=(None, None, 3), weights="imagenet", include_top=False)
    vgg = keras.Model(vgg.input, vgg.layers[20].output)

    if first_pass:
        discriminator = None
    else:
        discriminator = keras.saving.load_model()

    # Change the next line manually to switch pre-trained generators
    srresnet = keras.saving.load_model(paths.REPO_PATH + "/generators/srresnet-mse/srresnet-mse-e1000-lr0.001-resics45.keras")
    srgan = SRGAN(generator=srresnet, vgg=vgg, discriminator=discriminator)
    srgan.compile(d_optimiser=keras.optimizers.Adam(learning_rate=lr), g_optimiser=keras.optimizers.Adam(learning_rate=lr))
    srgan.fit(lr_dataset, hr_dataset, epochs=epochs, callbacks=[gan_saver])


if __name__ == "__main__":
    model = "srgan"
    lr = 10e-4
    epochs = 1
    details = {
        "model": model,
        "epochs": epochs,
        "lr": lr
    }
    
    downsample_factor = 4
    hr_dataset = np.load(paths.REPO_PATH + "/datasets/resics45_s2048.npy")
    lr_dataset = np.array([image[::downsample_factor, ::downsample_factor, :] for image in hr_dataset])

    if model.lower() == "srresnet-mse":
        train_srresnet_mse()
    if model.lower() == "srgan":
        train_srgan(first_pass=True)
