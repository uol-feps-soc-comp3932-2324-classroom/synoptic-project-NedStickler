
import keras
import numpy as np
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from models import SRResNet, SRGAN
from visualise import BlurAndResize
from loaders import load_resisc45_subset
import glob
import json
import pandas as pd


if __name__ == "__main__":
    ssim_dict = {}
    psnr_dict = {}
    hr_images, labels = load_resisc45_subset("test")

    
    for i in range(3, 45):
        hr_images = hr_images[i:i+45]
        lr_images = BlurAndResize(4)(hr_images)
        for path in glob.glob("generators/*"):
            model = path.split("/")[-1]
            ssim_df = pd.DataFrame()
            psnr_df = pd.DataFrame()

            if model == "srresnet-mse":
                generator = keras.saving.load_model("generators/srresnet-mse/srresnet-mse-e1588-resics45.keras")
                sr_images = generator(lr_images).numpy().astype(np.uint8)
                ssim = structural_similarity(hr_images, sr_images, channel_axis=3)
                psnr = peak_signal_noise_ratio(hr_images, sr_images)
                ssim_df.loc[0, model] += ssim
                ssim_df.loc[0, model] += psnr
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

                ssim_df.loc[0, model + "_1"] += fp_ssim
                ssim_df.loc[0, model + "_2"] += sp_ssim
                psnr_df.loc[0, model + "_1"] += fp_psnr
                psnr_df.loc[0, model + "_2"] += sp_psnr
    print(ssim_df)
    print(psnr_df)


        
