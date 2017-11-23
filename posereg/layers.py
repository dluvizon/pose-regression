# -*- coding: utf-8 -*-
from keras import backend as K

from keras.layers import Input
from keras.layers import Lambda
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers import SeparableConv2D
from keras.layers import BatchNormalization

from keras.layers import MaxPooling2D
from keras.layers import UpSampling2D

from keras.layers import multiply
from keras.layers import concatenate
from keras.layers import add

from posereg.math import linspace_2d
from posereg.activations import channel_softmax_2d


def conv(x, filters, size, strides=(1, 1), padding='same', name=None):
    x = Conv2D(filters, size, strides=strides, padding=padding,
            use_bias=False, name=name)(x)
    return x


def conv_bn(x, filters, size, strides=(1, 1), padding='same', name=None):
    if name is not None:
        conv_name = name + '_conv'
    else:
        conv_name = None

    x = conv(x, filters, size, strides, padding, conv_name)
    x = BatchNormalization(axis=-1, scale=False, name=name)(x)
    return x


def conv_bn_act(x, filters, size, strides=(1, 1), padding='same', name=None):
    if name is not None:
        conv_name = name + '_conv'
        bn_name = name + '_bn'
    else:
        conv_name = None
        bn_name = None

    x = conv(x, filters, size, strides, padding, conv_name)
    x = BatchNormalization(axis=-1, scale=False, name=bn_name)(x)
    x = Activation('relu', name=name)(x)
    return x


def act_conv_bn(x, filters, size, strides=(1, 1), padding='same', name=None):
    if name is not None:
        conv_name = name + '_conv'
        act_name = name + '_act'
    else:
        conv_name = None
        act_name = None

    x = Activation('relu', name=act_name)(x)
    x = conv(x, filters, size, strides, padding, conv_name)
    x = BatchNormalization(axis=-1, scale=False, name=name)(x)
    return x


def separable_act_conv_bn(x, filters, size, strides=(1, 1), padding='same',
        name=None):
    if name is not None:
        conv_name = name + '_conv'
        act_name = name + '_act'
    else:
        conv_name = None
        act_name = None

    x = Activation('relu', name=act_name)(x)
    x = SeparableConv2D(filters, size, strides=strides, padding=padding,
            use_bias=False, name=conv_name)(x)
    x = BatchNormalization(axis=-1, scale=False, name=name)(x)
    return x



def act_conv(x, filters, size, strides=(1, 1), padding='same', name=None):
    if name is not None:
        act_name = name + '_act'
    else:
        act_name = None

    x = Activation('relu', name=act_name)(x)
    x = conv(x, filters, size, strides, padding, name)
    return x


def act_channel_softmax(x, name=None):
    x = Activation(channel_softmax_2d(), name=name)(x)
    return x


def lin_interpolation_2d(inp, dim):

    num_rows, num_cols, num_filters = K.int_shape(inp)[1:]
    conv = SeparableConv2D(num_filters, (num_rows, num_cols), use_bias=False)
    x = conv(inp)

    w = conv.get_weights()
    w[0].fill(0)
    w[1].fill(0)
    linspace = linspace_2d(num_rows, num_cols, dim=dim)

    for i in range(num_filters):
        w[0][:,:, i, 0] = linspace[:,:]
        w[1][0, 0, i, i] = 1.

    conv.set_weights(w)
    conv.trainable = False

    x = Lambda(lambda x: K.squeeze(x, axis=1))(x)
    x = Lambda(lambda x: K.squeeze(x, axis=1))(x)
    x = Lambda(lambda x: K.expand_dims(x, axis=-1))(x)

    return x

