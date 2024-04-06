import keras
import tensorflow as tf
import cv2
import numpy as np
from keras import ops


@keras.saving.register_keras_serializable()
class PixelShuffle(keras.Layer):
    def call(self, x):
        return tf.nn.depth_to_space(x, 2)