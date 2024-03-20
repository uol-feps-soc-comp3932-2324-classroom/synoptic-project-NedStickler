import keras
from keras.callbacks import ModelCheckpoint
import numpy as np
from models import SRGAN, SRResNet
from utils import GANSaver
import paths


def train_srresnet_mse() -> None:
    save_checkpoint = ModelCheckpoint(paths.SAVE_PATH + f"/srresnet-mse/srresnet-mse-e{epochs}-resics45.keras", monitor="loss", save_best_only=True, mode="auto", save_freq="epoch")
    srresnet = SRResNet(16)
    srresnet.compile(optimizer=keras.optimizers.Adam(learning_rate=10**-4), loss=keras.losses.MeanSquaredError())
    srresnet.fit(lr_dataset, hr_dataset, epochs=epochs, callbacks=[save_checkpoint])
    

def train_srgan(first_pass: bool, vgg: int, discriminator_path: str = None, generator_path: str = None) -> None:
    if first_pass:
        discriminator = None
        generator = keras.saving.load_model(paths.REPO_PATH + "/generators/srresnet-mse/srresnet-mse-e1000-resics45.keras")
        lr = 10**-4
    else:
        discriminator = keras.saving.load_model(discriminator_path)
        generator = keras.saving.load_model(generator_path)
        lr = 10**-5

    details["lr"] = lr

    if vgg == 22:
        vgg_layer = 5
    elif vgg == 54:
        vgg_layer = 20

    gan_saver = GANSaver(paths.SAVE_PATH, details)
    vgg = keras.applications.VGG19(input_shape=(None, None, 3), weights="imagenet", include_top=False)
    vgg = keras.Model(vgg.input, vgg.layers[vgg_layer].output)

    srgan = SRGAN(generator=generator, vgg=vgg, discriminator=discriminator)
    srgan.compile(d_optimiser=keras.optimizers.Adam(learning_rate=lr), g_optimiser=keras.optimizers.Adam(learning_rate=lr))
    srgan.fit(lr_dataset, hr_dataset, epochs=epochs, callbacks=[gan_saver])


if __name__ == "__main__":
    downsample_factor = 4
    hr_dataset = np.load(paths.REPO_PATH + "/datasets/resics45_s2048.npy")
    lr_dataset = np.array([image[::downsample_factor, ::downsample_factor, :] for image in hr_dataset])

    model = "srgan-vgg54"
    epochs = 5

    details = {
        "model": model,
        "epochs": epochs,
    }
    
    if model == "srresnet-mse":
        train_srresnet_mse()
    elif model == "srgan-vgg22":
        discriminator_path = paths.REPO_PATH + "/generators/srgan-vgg22/srgan-vgg22-e100-lr0.0001-resics45/discriminator.keras"
        generator_path = paths.REPO_PATH + "/generators/srgan-vgg22/srgan-vgg22-e100-lr0.0001-resics45/generator.keras"
        train_srgan(first_pass=False, vgg=22, discriminator_path=discriminator_path, generator_path=generator_path)
    elif model == "srgan-vgg54":
        discriminator_path = paths.REPO_PATH + "/generators/srgan-vgg54/srgan-vgg54-e100-lr0.0001-resics45/discriminator.keras"
        generator_path = paths.REPO_PATH + "/generators/srgan-vgg54/srgan-vgg54-e100-lr0.0001-resics45/generator.keras"
        train_srgan(first_pass=False, vgg=54, discriminator_path=discriminator_path, generator_path=generator_path)
