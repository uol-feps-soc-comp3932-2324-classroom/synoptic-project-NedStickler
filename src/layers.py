import keras
import tensorflow as tf


@keras.saving.register_keras_serializable(name="pixel_shuffle")
class PixelShuffle(keras.Layer):
    def call(self, x):
        return tf.nn.depth_to_space(x, 2)
    
    def build(self, input_shape):
        super(PixelShuffle, self).build(input_shape) 
