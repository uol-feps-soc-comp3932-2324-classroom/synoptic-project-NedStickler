
import keras
import numpy as np
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from models import SRResNet, SRGAN
from visualise import BlurAndResize
from loaders import load_resisc45_subset
import glob


if __name__ == "__main__":
    ssim_dict = {}
    psnr_dict = {}
    hr_images, labels = load_resisc45_subset("test")
    hr_images = hr_images[:45]
    lr_images = BlurAndResize(4)(hr_images)

    for path in glob.glob("generators/*"):
        model = path.split("/")[-1] 
        if model == "srresnet-mse":
            generator = keras.saving.load_model("generators/srresnet-mse/srresnet-mse-e1588-resics45.keras")
            sr_images = generator(lr_images).numpy().astype(np.uint8)
            ssim = structural_similarity(hr_images, sr_images, channel_axis=3)
            psnr = peak_signal_noise_ratio(hr_images, sr_images)
            ssim_dict[model] = ssim
            psnr_dict[model] = psnr
        else:
            fp_path = path + f"/{model}-e159-lr0.0001-resics45/generator.keras"
            sp_path = path + f"/{model}-e159-lr1e-05-resics45/generator.keras"

            fp_generator = keras.saving.load_model(fp_path)
            sp_generator = keras.saving.load_model(fp_path)

            fp_sr_images = fp_generator(lr_images).numpy().astype(np.uint8)
            sp_sr_images = sp_generator(lr_images).numpy().astype(np.uint8)

            fp_ssim = structural_similarity(hr_images, fp_sr_images, channel_axis=3)
            sp_ssim = structural_similarity(hr_images, sp_sr_images, channel_axis=3)
            fp_psnr = peak_signal_noise_ratio(hr_images, fp_sr_images)
            sp_psnr = peak_signal_noise_ratio(hr_images, sp_sr_images)

            ssim_dict[model + "_1"] = fp_ssim
            ssim_dict[model + "_2"] = sp_ssim
            psnr_dict[model + "_1"] = fp_psnr
            psnr_dict[model + "_2"] = sp_psnr

    print(ssim_dict)
    print(psnr_dict)


        
