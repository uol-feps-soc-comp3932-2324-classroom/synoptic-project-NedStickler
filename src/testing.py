
import keras
import numpy as np
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from models import SRResNet, SRGAN
from visualise import BlurAndResize
from loaders import load_resisc45_subset, load_set5, load_set14
import glob
import json
import pandas as pd
import cv2

def main():
    resisc45_hr, labels = load_resisc45_subset("test")
    generators = [generator.split("/")[-1] for generator in glob.glob("generators/*")]

    columns = []
    generators_map = {}
    for generator in generators:
        if generator != "srresnet-mse":
            columns.append(generator + "_1")
            columns.append(generator + "_2")
            fp_g = keras.saving.load_model(f"generators/{generator}/{generator}-e159-lr0.0001-resisc45/generator.keras")
            sp_g= keras.saving.load_model(f"generators/{generator}/{generator}-e159-lr1e-05-resisc45/generator.keras")
            generators_map[generator + "_1"] = fp_g
            generators_map[generator + "_2"] = sp_g
        else:
            columns.append(generator)
            g = keras.saving.load_model("generators/srresnet-mse/srresnet-mse-e1588-resisc45.keras")
            generators_map["srresnet-mse"] = g

    ssim_df = pd.DataFrame(0.0, index=["nwpu-resisc45", "set5", "set14"], columns=columns)
    psnr_df = pd.DataFrame(0.0, index=["nwpu-resisc45", "set5", "set14"], columns=columns)
    
    for i in range(0, 135, 45):
        hr = resisc45_hr[i:i+45]
        lr = BlurAndResize(4)(hr)
        for generator in generators:
            if generator == "srresnet-mse":
                g = generators_map["srresnet-mse"]
                sr = g(lr).numpy().astype(np.uint8)
                ssim = structural_similarity(hr, sr, channel_axis=3)
                psnr = peak_signal_noise_ratio(hr, sr)
                ssim_df.loc["nwpu-resisc45", generator] += ssim
                psnr_df.loc["nwpu-resisc45", generator] += psnr
            else:
                fp_g = generators_map[generator + "_1"]
                sp_g = generators_map[generator + "_2"]

                fp_sr = fp_g(lr).numpy().astype(np.uint8)
                sp_sr = sp_g(lr).numpy().astype(np.uint8)

                fp_ssim = structural_similarity(hr, fp_sr, channel_axis=3)
                sp_ssim = structural_similarity(hr, sp_sr, channel_axis=3)
                fp_psnr = peak_signal_noise_ratio(hr, fp_sr)
                sp_psnr = peak_signal_noise_ratio(hr, sp_sr)

                ssim_df.loc["nwpu-resisc45", generator + "_1"] += fp_ssim
                ssim_df.loc["nwpu-resisc45", generator + "_2"] += sp_ssim
                psnr_df.loc["nwpu-resisc45", generator + "_1"] += fp_psnr
                psnr_df.loc["nwpu-resisc45", generator + "_2"] += sp_psnr
    
    ssim_df /= 3
    psnr_df /= 3

    set5_lr, set5_hr = load_set5()
    set14_lr, set14_hr = load_set14()
    for generator in generators_map.keys():
        g = generators_map[generator]
        set5_sr = [g(np.array([img])).numpy().astype(np.uint8)[0] for img in set5_lr]
        set5_ssim = np.mean([structural_similarity(hr, sr, channel_axis=2) for hr, sr in zip(set5_hr, set5_sr)])
        set5_psnr = np.mean([peak_signal_noise_ratio(hr, sr) for hr, sr in zip(set5_hr, set5_sr)])

        set14_sr = [g(np.array([img])).numpy().astype(np.uint8)[0] for img in set14_lr]
        set14_ssim = np.mean([structural_similarity(hr, sr, channel_axis=2) for hr, sr in zip(set14_hr, set14_sr)])
        set14_psnr = np.mean([peak_signal_noise_ratio(hr, sr) for hr, sr in zip(set14_hr, set14_sr)])

        ssim_df.loc["set5", generator] = set5_ssim
        ssim_df.loc["set14", generator] = set14_ssim
        psnr_df.loc["set5", generator] = set5_psnr
        psnr_df.loc["set14", generator] = set14_psnr
    
    ssim_df.to_csv("/uolstore/home/users/sc20ns/Documents/synoptic-project-NedStickler/datasets/ssim.csv")
    psnr_df.to_csv("/uolstore/home/users/sc20ns/Documents/synoptic-project-NedStickler/datasets/psnr.csv")


def bicubic():
    resisc_hr = np.load(r"C:\Users\nedst\Desktop\synoptic-project-NedStickler\datasets\resisc45_test.npy")
    resisc_lr = BlurAndResize(4)(resisc_hr).numpy().astype(np.uint8)

    set5_lr = []
    set5_hr = []
    for file in glob.glob(r"C:\Users\nedst\Desktop\SelfExSR\data\Set5\image_SRF_4\*.png"):
        img = cv2.imread(file)
        if file.split(".png")[0][-2:] == "HR":
            set5_hr.append(img)
        else:
            set5_lr.append(img)
    
    set14_lr = []
    set14_hr = []
    for file in glob.glob(r"C:\Users\nedst\Desktop\SelfExSR\data\Set14\image_SRF_4\*.png"):
        img = cv2.imread(file)
        if file.split(".png")[0][-2:] == "HR":
            set14_hr.append(img)
        else:
            set14_lr.append(img)
    
    resisc45_bicubic = keras.layers.Resizing(256, 256, "bicubic")
    resisc45_sr = resisc45_bicubic(resisc_lr).numpy().astype(np.uint8)
    resisc45_ssim = structural_similarity(resisc_hr, resisc45_sr, channel_axis=3)
    resisc45_psnr = peak_signal_noise_ratio(resisc_hr, resisc45_sr)

    set5_sr = []
    for i, images in enumerate(zip(set5_lr, set5_hr)):
        lr, hr = images
        set5_bicubic = keras.layers.Resizing(hr.shape[0], hr.shape[1], interpolation="bicubic")
        set5_sr.append(set5_bicubic(lr).numpy().astype(np.uint8))
    set5_ssim = np.mean([structural_similarity(hr, sr, channel_axis=2) for hr, sr in zip(set5_hr, set5_sr)])
    set5_psnr = np.mean([peak_signal_noise_ratio(hr, sr) for hr, sr in zip(set5_hr, set5_sr)])

    set14_sr = []
    for i, images in enumerate(zip(set14_lr, set14_hr)):
        lr, hr = images
        set14_bicubic = keras.layers.Resizing(hr.shape[0], hr.shape[1], interpolation="bicubic")
        set14_sr.append(set14_bicubic(lr).numpy().astype(np.uint8))
    set14_ssim = np.mean([structural_similarity(hr, sr, channel_axis=2) for hr, sr in zip(set14_hr, set14_sr)])
    set14_psnr = np.mean([peak_signal_noise_ratio(hr, sr) for hr, sr in zip(set14_hr, set14_sr)])

    print(resisc45_ssim)
    print(set5_ssim)
    print(set14_ssim)
    print(resisc45_psnr)
    print(set5_psnr)
    print(set14_psnr)

    
if __name__ == "__main__":
    # main()
    bicubic()