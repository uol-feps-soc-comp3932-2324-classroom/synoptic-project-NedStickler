import keras
import tensorflow as tf
import cv2
import numpy as np
from keras import ops


@keras.saving.register_keras_serializable()
class PixelShuffle(keras.Layer):
    def call(self, x):
        return tf.nn.depth_to_space(x, 2)


@keras.saving.register_keras_serializable()
class GaussianBlur(keras.Layer):
    def call(self, x):
        x = ops.unstack(x)
        x = [cv2.GaussianBlur(img.numpy().astype(np.uint8), (7, 7), 0) for img in x]
        x = ops.stack(x)
        return x