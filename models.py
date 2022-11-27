import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import time
import matplotlib.pyplot as plt

# import keras
import tensorflow as tf
from tensorflow.keras.layers import Input, Reshape, Dropout, Dense, concatenate
from tensorflow.keras.layers import Flatten, BatchNormalization
from tensorflow.keras.layers import Activation, ZeroPadding2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras import losses
# from tensorflow.keras.layers import merge
# from tensorflow.keras.layers.advanced_activations import LeakyReLU
# from tensorflow.keras.layers.convolutional import Convolution2D, Deconvolution2D
# from tensorflow.keras.layers.core import Activation, Dropout
# from tensorflow.keras.layers.normalization import BatchNormalization
from tensorflow.keras.models import Model

# from tensorflow.keras.optimizers import Adam


KERAS_2 = tf.keras.__version__[0] == '2'

try:
    # keras 2 imports
    from keras.layers.convolutional import Conv2DTranspose
    from keras.layers.merge import Concatenate
except ImportError:
    print("Keras 2 layers could not be imported defaulting to keras1")
    KERAS_2 = False


def concatenate_layers(inputs, concat_axis, mode='concat'):
    if KERAS_2:
        assert mode == 'concat', "Only concatenation is supported in this wrapper"
        return Concatenate(axis=concat_axis)(inputs)
    else:
        return concatenate(inputs=inputs, concat_axis=concat_axis, mode=mode)

def Convolution(f, k=3, s=2, border_mode='same', **kwargs):
    """Convenience method for Convolutions."""
    if KERAS_2:
        return Conv2D(f,
                             kernel_size=(k, k),
                             padding=border_mode,
                             strides=(s, s),
                             **kwargs)
    else:
        return Conv2D(f, k, k, border_mode=border_mode,
                             subsample=(s, s),
                             **kwargs)

def Deconvolution(f, output_shape, k=2, s=2, **kwargs):
    """Convenience method for Transposed Convolutions."""
    if KERAS_2:
        return Conv2DTranspose(f, kernel_size=(k, k),
                               output_shape=output_shape,
                               strides=(s, s),
                               data_format=K.image_data_format(),
                               **kwargs)
    else:
        return Conv2DTranspose(f, k, k, output_shape=output_shape,
                               subsample=(s, s), **kwargs)


def BatchNorm(mode=2, axis=1, **kwargs):
    """Convenience method for BatchNormalization layers."""
    if KERAS_2:
        return BatchNormalization(axis=axis, **kwargs)
    else:
        return BatchNormalization(mode=2,axis=axis, **kwargs)
                             


def g_unet(in_ch, out_ch, nf, batch_size=1, is_binary=False, name='unet'):
    merge_params = {
        'mode': 'concat',
        'concat_axis': 1
    }
    if K.image_data_format() == 'th':
        print('TheanoShapedU-NET')
        i = Input(shape=(in_ch, 512, 512))

        def get_deconv_shape(samples, channels, x_dim, y_dim):
            return samples, channels, x_dim, y_dim

    elif K.image_data_format() == 'tf':
        i = Input(shape=(512, 512, in_ch))
        print('TensorflowShapedU-NET')

        def get_deconv_shape(samples, channels, x_dim, y_dim):
            return samples, x_dim, y_dim, channels

        merge_params['concat_axis'] = 3
    else:
        raise ValueError(
            'Keras dimension ordering not supported: {}'.format(
                K.image_data_format()))
    
    # in_ch x 512 x 512
    conv1 = Convolution(nf)(i)
    conv1 = BatchNorm()(conv1)
    x = LeakyReLU(0.2)(conv1)
    # nf x 256 x 256

    conv2 = Convolution(nf * 2)(x)
    conv2 = BatchNorm()(conv2)
    x = LeakyReLU(0.2)(conv2)
    # nf*2 x 128 x 128

    conv3 = Convolution(nf * 4)(x)
    conv3 = BatchNorm()(conv3)
    x = LeakyReLU(0.2)(conv3)
    # nf*4 x 64 x 64

    conv4 = Convolution(nf * 8)(x)
    conv4 = BatchNorm()(conv4)
    x = LeakyReLU(0.2)(conv4)
    # nf*8 x 32 x 32

    conv5 = Convolution(nf * 8)(x)
    conv5 = BatchNorm()(conv5)
    x = LeakyReLU(0.2)(conv5)
    # nf*8 x 16 x 16

    conv6 = Convolution(nf * 8)(x)
    conv6 = BatchNorm()(conv6)
    x = LeakyReLU(0.2)(conv6)
    # nf*8 x 8 x 8

    conv7 = Convolution(nf * 8)(x)
    conv7 = BatchNorm()(conv7)
    x = LeakyReLU(0.2)(conv7)
    # nf*8 x 4 x 4

    conv8 = Convolution(nf * 8)(x)
    conv8 = BatchNorm()(conv8)
    x = LeakyReLU(0.2)(conv8)
    # nf*8 x 2 x 2

    conv9 = Convolution(nf * 8, k=2, s=1, border_mode='valid')(x)
    conv9 = BatchNorm()(conv9)
    x = LeakyReLU(0.2)(conv9)
    # nf*8 x 1 x 1
    
    dconv1 = Deconvolution(nf * 8, get_deconv_shape(batch_size, nf * 8, 2, 2),
                           k=2, s=1)(x)
    dconv1 = BatchNorm()(dconv1)
    dconv1 = Dropout(0.5)(dconv1)

    x = concatenate_layers([dconv1, conv8], **merge_params)

    x = LeakyReLU(0.2)(x)
    # nf*(8 + 8) x 2 x 2

    dconv2 = Deconvolution(nf * 8, get_deconv_shape(batch_size, nf * 8, 4, 4))(x)
    dconv2 = BatchNorm()(dconv2)
    dconv2 = Dropout(0.5)(dconv2)
    x = concatenate_layers([dconv2, conv7], **merge_params)
    x = LeakyReLU(0.2)(x)
    # nf*(8 + 8) x 4 x 4

    dconv3 = Deconvolution(nf * 8, get_deconv_shape(batch_size, nf * 8, 8, 8))(x)
    dconv3 = BatchNorm()(dconv3)
    dconv3 = Dropout(0.5)(dconv3)
    x = concatenate_layers([dconv3, conv6], **merge_params)
    x = LeakyReLU(0.2)(x)
    # nf*(8 + 8) x 8 x 8

    dconv4 = Deconvolution(nf * 8, get_deconv_shape(batch_size, nf * 8, 16, 16))(x)
    dconv4 = BatchNorm()(dconv4)
    x = concatenate_layers([dconv4, conv5], **merge_params)
    x = LeakyReLU(0.2)(x)
    # nf*(8 + 8) x 16 x 16

    dconv5 = Deconvolution(nf * 8, get_deconv_shape(batch_size, nf * 8, 32, 32))(x)
    dconv5 = BatchNorm()(dconv5)
    x = concatenate_layers([dconv5, conv4], **merge_params)
    x = LeakyReLU(0.2)(x)
    # nf*(8 + 8) x 32 x 32

    dconv6 = Deconvolution(nf * 4, get_deconv_shape(batch_size, nf * 4, 64, 64))(x)
    dconv6 = BatchNorm()(dconv6)
    x = concatenate_layers([dconv6, conv3], **merge_params)
    x = LeakyReLU(0.2)(x)
    # nf*(4 + 4) x 64 x 64

    dconv7 = Deconvolution(nf * 2, get_deconv_shape(batch_size, nf * 2, 128, 128))(x)
    dconv7 = BatchNorm()(dconv7)
    x = concatenate_layers([dconv7, conv2], **merge_params)
    x = LeakyReLU(0.2)(x)
    # nf*(2 + 2) x 128 x 128

    dconv8 = Deconvolution(nf, get_deconv_shape(batch_size, nf, 256, 256))(x)
    dconv8 = BatchNorm()(dconv8)
    x = concatenate_layers([dconv8, conv1], **merge_params)
    x = LeakyReLU(0.2)(x)
    # nf*(1 + 1) x 256 x 256

    dconv9 = Deconvolution(out_ch, get_deconv_shape(batch_size, out_ch, 512, 512))(x)
    # out_ch x 512 x 512

    act = 'sigmoid' if is_binary else 'tanh'
    out = Activation(act)(dconv9)

    unet = Model(i, out, name=name)

    return unet

def discriminator(a_ch, b_ch, nf, opt=Adam(lr=2e-4, beta_1=0.5), name='d'):
    i = Input(shape=(a_ch + b_ch, 512, 512))

    # (a_ch + b_ch) x 512 x 512
    conv1 = Convolution(nf)(i)
    x = LeakyReLU(0.2)(conv1)
    # nf x 256 x 256

    conv2 = Convolution(nf * 2)(x)
    x = LeakyReLU(0.2)(conv2)
    # nf*2 x 128 x 128

    conv3 = Convolution(nf * 4)(x)
    x = LeakyReLU(0.2)(conv3)
    # nf*4 x 64 x 64

    conv4 = Convolution(nf * 8)(x)
    x = LeakyReLU(0.2)(conv4)
    # nf*8 x 32 x 32

    conv5 = Convolution(1)(x)
    out = Activation('sigmoid')(conv5)
    # 1 x 16 x 16

    d = Model(i, out, name=name)

    def d_loss(y_true, y_pred):
        L = losses.binary_crossentropy(K.batch_flatten(y_true), K.batch_flatten(y_pred))
        return L

    d.compile(optimizer=opt, loss=d_loss)
    return d


def pix2pix(atob, d, a_ch, b_ch, alpha=100, is_a_binary=False, is_b_binary=False, opt=Adam(lr=2e-4, beta_1=0.5), name='pix2pix'):
    a = Input(shape=(a_ch, 512, 512))
    b = Input(shape=(b_ch, 512, 512))

    # A -> B'
    bp = atob(a)

    # Discriminator receives the pair of images
    d_in = concatenate_layers([a, bp], mode='concat', concat_axis=1)

    pix2pix = Model([a, b], d(d_in), name=name)

    def pix2pix_loss(y_true, y_pred):
        y_true_flat = K.batch_flatten(y_true)
        y_pred_flat = K.batch_flatten(y_pred)

        # Adversarial Loss
        L_adv = losses.binary_crossentropy(y_true_flat, y_pred_flat)

        # A to B loss
        b_flat = K.batch_flatten(b)
        bp_flat = K.batch_flatten(bp)
        if is_b_binary:
            L_atob = losses.binary_crossentropy(b_flat, bp_flat)
        else:
            L_atob = K.mean(K.abs(b_flat - bp_flat))

        return L_adv + alpha * L_atob

    # This network is used to train the generator. Freeze the discriminator part.
    pix2pix.get_layer('d').trainable = False

    pix2pix.compile(optimizer=opt, loss=pix2pix_loss)
    return pix2pix

if __name__ == '__main__':
    import doctest

    TEST_TF = True
    if TEST_TF:
        os.environ['KERAS_BACKEND'] = 'tensorflow'
    else:
        os.environ['KERAS_BACKEND'] = 'theano'
    doctest.testsource('models.py', verbose=True, optionflags=doctest.ELLIPSIS)