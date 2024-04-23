
import keras
from keras.layers import Resizing, DepthwiseConv2D
import numpy as np
from models import SRGAN, SRResNet
from utils import visualise_generator


class BlurAndResize(keras.Model):
    def __init__(self, downsample_factor):
        super().__init__()
        self.downsample_factor = downsample_factor
        self.gaussian_blur = DepthwiseConv2D(3, padding="same", use_bias=False)
        self.gaussian_blur.build((256, 256, 3))
        self.resize = Resizing(256 // downsample_factor, 256 // downsample_factor, interpolation="bicubic")

    def _matlab_style_gauss2D(self, shape=(3,3), sigma=0.5):
        """
        2D gaussian mask - should give the same result as MATLAB's
        fspecial('gaussian',[shape],[sigma])
        """
        m,n = [(ss-1.)/2. for ss in shape]
        y,x = np.ogrid[-m:m+1,-n:n+1]
        h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
        h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
        sumh = h.sum()
        if sumh != 0:
            h /= sumh
        return h
    
    def get_config(self):
        return {"downsample_factor": self.downsample_factor}
    
    def call(self, inputs):
        gaussian_kernel = self._matlab_style_gauss2D()
        kernel_weights = np.expand_dims(gaussian_kernel, axis=-1)
        kernel_weights = np.repeat(kernel_weights, 3, axis=-1)
        kernel_weights = np.expand_dims(kernel_weights, axis=-1)
        self.gaussian_blur.set_weights([kernel_weights])
        self.gaussian_blur.trainable = False
        blurred_hr = self.gaussian_blur(inputs)
        lr_patch = self.resize(blurred_hr)
        return lr_patch


if __name__ == "__main__":
    loss = "efficientnetv2l"
    generator_path = f"C:\\Users\\nedst\\Desktop\\synoptic-project-NedStickler\\generators\\srgan-{loss}\\srgan-{loss}-e159-lr1e-05-resisc45\\generator.keras"
    hr_images = np.load(r"C:\Users\nedst\Desktop\synoptic-project-NedStickler\datasets\resisc45_test.npy")
    lr_images = BlurAndResize(4)(hr_images).numpy().astype(np.uint8)
    generator = keras.saving.load_model(generator_path)
    visualise_generator(generator, lr_images[:3], hr_images[:3])