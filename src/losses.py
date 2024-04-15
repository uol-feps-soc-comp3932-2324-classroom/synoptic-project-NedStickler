
import keras
import numpy as np
from models import CustomLoss
from loaders import load_resisc45_subset


class Losses:
    def __init__(self):
        self.input_shape = (None, None, 3)
    
    def vgg19(self, feature_level):
        if feature_level == 54:
            layer = 20
        else:
            layer = 5
        vgg19 = keras.applications.VGG19(include_top=False, input_shape=self.input_shape)
        vgg19 = keras.Model(vgg19.input, vgg19.layers[layer].output)
        vgg19_preprocess = keras.applications.vgg19.preprocess_input
        return CustomLoss(vgg19, vgg19_preprocess, scale=1/12.75)
    
    def xception(self):
        xception = keras.applications.Xception(include_top=False, input_shape=self.input_shape)
        xception_preprocess = keras.applications.xception.preprocess_input
        return CustomLoss(xception, xception_preprocess, scale=1/0.56)
    
    def resnet152v2(self):
        resnet152v2 = keras.applications.ResNet152V2(include_top=False, input_shape=self.input_shape)
        resnet152v2_preprocess = keras.applications.resnet_v2.preprocess_input
        return CustomLoss(resnet152v2, resnet152v2_preprocess, scale=1/3.51)
    
    def inceptionv3(self):
        inceptionv3 = keras.applications.InceptionV3(include_top=False, input_shape=self.input_shape)
        inceptionv3_preprocess = keras.applications.inception_v3.preprocess_input
        return CustomLoss(inceptionv3, inceptionv3_preprocess, scale=1/0.84)
    
    def inceptionresnetv2(self):
        inceptionresnetv2 = keras.applications.InceptionResNetV2(include_top=False, input_shape=self.input_shape)
        inceptionresnetv2_preprocess = keras.applications.inception_resnet_v2.preprocess_input
        return CustomLoss(inceptionresnetv2, inceptionresnetv2_preprocess, scale=1/0.91)
    
    def mobilenetv2(self):
        mobilenetv2 = keras.applications.MobileNetV2(include_top=False, input_shape=self.input_shape)
        mobilenetv2_preprocess = keras.applications.mobilenet_v2.preprocess_input
        return CustomLoss(mobilenetv2, mobilenetv2_preprocess, scale=1/2.34)
    
    def densenet201(self):
        densenet201 = keras.applications.DenseNet201(include_top=False, input_shape=self.input_shape)
        densenet201_preprocess = keras.applications.densenet.preprocess_input
        return CustomLoss(densenet201, densenet201_preprocess, scale=1/0.42)
    
    def nasnetlarge(self):
        nasnetlarge = keras.applications.NASNetLarge(include_top=False, input_shape=(96, 96, 3))
        nasnetlarge_preprocess = keras.applications.nasnet.preprocess_input
        return CustomLoss(nasnetlarge, nasnetlarge_preprocess, scale=1/1.56)

    def efficientnetv2l(self):
        efficientnetv2l = keras.applications.EfficientNetV2L(include_top=False, input_shape=self.input_shape)
        efficientnetv2l_preprocess = keras.applications.efficientnet_v2.preprocess_input
        return CustomLoss(efficientnetv2l, efficientnetv2l_preprocess, scale=1/0.003)
    

if __name__ == "__main__":
    validation_data, _ = load_resisc45_subset("val")
    validation_data = validation_data[:15]
    losses = Losses()
    loss = losses.efficientnetv2l()
    output = loss(validation_data)
    average = np.mean([val for val in output.numpy().flatten() if val != 0])
    print(average)


