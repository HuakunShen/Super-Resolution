# https://arxiv.org/abs/1612.07919

# pip install tensorflow
# pip install keras
from keras.applications.vgg19 import VGG19, preprocess_input
from tensorflow.keras.layers import Flatten
from keras import backend as K

import cv2 as cv, numpy as np, numpy.linalg as LA

vgg_model = VGG19()
block2_pool = vgg_model.get_layer("block2_pool").output
f2 = Flatten()(block2_pool)
func2 = K.function([vgg_model.input], [f2])

block5_pool = vgg_model.get_layer("block5_pool").output
f5 = Flatten()(block5_pool)
func5 = K.function([vgg_model.input], [f5])

def loss_mse(img1, img2):
    weight = 224**2 / (img1.shape[0]*img1.shape[1])
    return weight * LA.norm(img1 - img2)

def loss_perceptual(img1, img2):
    img1_pre = cv.resize(img1, (224,224), interpolation=cv.INTER_AREA)
    img1_pre = np.expand_dims(img1_pre, axis=0)
    img1_pre = preprocess_input(img1_pre)
    
    img1_pool2 = func2(img1_pre)[0]
    img1_pool5 = func5(img1_pre)[0]
    
    img2_pre = cv.resize(img2, (224,224), interpolation=cv.INTER_AREA)
    img2_pre = np.expand_dims(img2_pre, axis=0)
    img2_pre = preprocess_input(img2_pre)
    
    img2_pool2 = func2(img2_pre)[0]
    img2_pool5 = func5(img2_pre)[0]
    
    return LA.norm(img1_pool2 - img2_pool2), LA.norm(img1_pool5 - img2_pool5)

def loss_sum(img1, img2):
    mse = loss_mse(img1, img2)
    pool2, pool5 = loss_perceptual(img1, img2)
    return mse + pool2 + pool5
