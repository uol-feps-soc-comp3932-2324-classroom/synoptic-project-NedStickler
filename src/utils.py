import matplotlib.pyplot as plt
import numpy as np


def visualise_generator(generator, lr_imgs, hr_imgs) -> None:
    sr_imgs = generator(lr_imgs)
    num_sr_imgs = len(sr_imgs)

    fig, axs = plt.subplots(num_sr_imgs, 3)
    fig.set_size_inches(9, num_sr_imgs * 3)

    axs[0, 0].set_title("LR")
    axs[0, 1].set_title("HR")
    axs[0, 2].set_title("SR")

    for i, img in enumerate(sr_imgs):
        axs[i, 0].imshow(lr_imgs[i])
        axs[i, 1].imshow(hr_imgs[i])
        axs[i, 2].imshow(img.numpy().astype(np.uint8))

    plt.show()