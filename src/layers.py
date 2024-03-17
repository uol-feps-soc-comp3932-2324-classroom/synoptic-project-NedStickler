import keras
from tensorflow.nn import depth_to_space


@keras.saving.register_keras_serializable()
class PixelShuffle(keras.Layer):
    def call(self, x):
        return depth_to_space(x, 2)
    
    def get_config(self):
        return {}
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)