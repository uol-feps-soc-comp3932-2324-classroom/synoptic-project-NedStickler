
import keras
from models import CustomLoss


class Losses:
    def __init__(self):
        self.input_shape = (256, 256, 3)
    
    def vgg19(self, feature_level):
        if feature_level == 54:
            layer = 20
        else:
            layer = 5
        vgg19 = keras.applications.VGG19(include_top=False, input_shape=self.input_shape)
        vgg19 = keras.Model(vgg19.input, vgg19[layer].output)
        vgg19_preprocess = keras.applications.vgg19.preprocess_input
        return CustomLoss(vgg19, vgg19_preprocess, scale=1/12.75)
    
    def xception(self):
        xception = keras.applications.Xception(include_top=False, input_shape=self.input_shape)
        xception_preprocess = keras.applications.xception.preprocess_input
        return CustomLoss(xception, xception_preprocess, scale=1)
    
    def resnet152v2(self):
        resnet152v2 = keras.applications.ResNet152V2(include_top=False, input_shape=self.input_shape)
        resnet152v2_preprocess = keras.applications.resnet_v2.preprocess_input
        return CustomLoss(resnet152v2, resnet152v2_preprocess, scale=1)
    
    def inceptionv3(self):
        inceptionv3 = keras.applications.InceptionV3(include_top=False, input_shape=self.input_shape)
        inceptionv3_preprocess = keras.applications.inception_v3.preprocess_input
        return CustomLoss(inceptionv3, inceptionv3_preprocess, scale=1)
    
    def inceptionresnetv2(self):
        inceptionresnetv2 = keras.applications.InceptionResNetV2(include_top=False, input_shape=self.input_shape)
        inceptionresnetv2_preprocess = keras.applications.inception_resnet_v2.preprocess_input
        return CustomLoss(inceptionresnetv2, inceptionresnetv2_preprocess, scale=1)
    
    def mobilenetv2(self):
        mobilenetv2 = keras.applications.MobileNetV2(include_top=False, input_shape=self.input_shape)
        mobilenetv2_preprocess = keras.applications.mobilenet_v2.preprocess_input
        return CustomLoss(mobilenetv2, mobilenetv2_preprocess, scale=1)
    
    def densenet201(self):
        densenet201 = keras.applications.DenseNet201(include_top=False, input_shape=self.input_shape)
        densenet201_preprocess = keras.applications.densenet.preprocess_input
        return CustomLoss(densenet201, densenet201_preprocess, scale=1)
    
    def nasnetlarge(self):
        nasnetlarge = keras.applications.NASNetLarge(include_top=False, input_shape=self.input_shape)
        nasnetlarge_preprocess = keras.applications.nasnet.preprocess_input
        return CustomLoss(nasnetlarge, nasnetlarge_preprocess, scale=1)
        