import keras
import tensorflow as tf


@keras.saving.register_keras_serializable()
class PixelShuffle(keras.Layer):
    def call(self, x):
        return tf.nn.depth_to_space(x, 2)
    
    def get_config(self):
        return {}
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)