# -*- coding: utf-8 -*-
"""Define the ReceptionNet for human pose estimation for Keras and TensorFlow.

The network is defined as:

-------   ------
|Input|-->|Stem|--> [...],
-------   ------

end every prediction block:

                     -----------------------------------------------
                     |             --------------------------------|
           --------- |  ---------- |  ---------      ---------     |
    [...]->|rBlockN|--->|SepConvN|--->|RegMapN|-(H)->|fReMapN|--->(+)-->[...]
           ---------    ----------    ---------      ---------


                  |-->(sSAM)-------------------
         |--(Hs)--|                           |
         |        |-->(sjProp)--> *visible*   |
    H -> |                                    |
         |        |-->(cSAM)----------------(Agg)--> *pose*
         |--(Hc)--|                           |
                  |-->(cjProp)----------------|
"""
from keras.models import Model
from keras.optimizers import RMSprop

# Needs tf.divide, which is not implemented in Keras backend
import tensorflow as tf

from posereg.objectives import elasticnet_loss_on_valid_joints
from posereg.layers import *


def sepconv_residual(x, out_size, name, kernel_size=(3, 3)):
    shortcut_name = name + '_shortcut'
    reduce_name = name + '_reduce'

    num_filters = K.int_shape(x)[-1]
    if num_filters == out_size:
        ident = x
    else:
        ident = act_conv_bn(x, out_size, (1, 1), name=shortcut_name)

    if out_size < num_filters:
        x = act_conv_bn(x, out_size, (1, 1), name=reduce_name)

    x = separable_act_conv_bn(x, out_size, kernel_size, name=name)
    x = add([ident, x])

    return x


def stem(inp):
    xi = Input(shape=K.int_shape(inp)[1:]) # Expected 256 x 256 x 3

    x = conv_bn_act(xi, 32, (3, 3), strides=(2, 2))
    x = conv_bn_act(x, 32, (3, 3))
    x = conv_bn_act(x, 64, (3, 3))

    a = conv_bn_act(x, 96, (3, 3), strides=(2, 2))
    b = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = concatenate([a, b])

    a = conv_bn_act(x, 64, (1, 1))
    a = conv_bn(a, 96, (3, 3))
    b = conv_bn_act(x, 64, (1, 1))
    b = conv_bn_act(b, 64, (5, 1))
    b = conv_bn_act(b, 64, (1, 5))
    b = conv_bn(b, 96, (3, 3))
    x = concatenate([a, b])

    a = act_conv_bn(x, 192, (3, 3), strides=(2, 2))
    b = MaxPooling2D((2, 2), strides=(2, 2))(x)
    x = concatenate([a, b])

    x = sepconv_residual(x, 3*192, name='sepconv1')

    model = Model(xi, x, name='Stem')
    x = model(inp)

    return x


def build_reception_block(inp, name, ksize=(3, 3)):
    input_shape = K.int_shape(inp)[1:]
    size = input_shape[-1]

    xi = Input(shape=input_shape)
    a = sepconv_residual(xi, size, name='sepconv_l1', kernel_size=ksize)

    low1 = MaxPooling2D((2, 2))(xi)
    low1 = act_conv_bn(low1, int(size/2), (1, 1))
    low1 = sepconv_residual(low1, int(size/2), name='sepconv_l2_1',
            kernel_size=ksize)
    b = sepconv_residual(low1, int(size/2), name='sepconv_l2_2',
            kernel_size=ksize)

    c = MaxPooling2D((2, 2))(low1)
    c = sepconv_residual(c, int(size/2), name='sepconv_l3_1',
            kernel_size=ksize)
    c = sepconv_residual(c, int(size/2), name='sepconv_l3_2',
            kernel_size=ksize)
    c = sepconv_residual(c, int(size/2), name='sepconv_l3_3',
            kernel_size=ksize)
    c = UpSampling2D((2, 2))(c)

    b = add([b, c])
    b = sepconv_residual(b, size, name='sepconv_l2_3', kernel_size=ksize)
    b = UpSampling2D((2, 2))(b)
    x = add([a, b])

    model = Model(inputs=xi, outputs=x, name=name)

    return model(inp)


def build_sconv_block(inp, name=None, ksize=(3, 3)):
    input_shape = K.int_shape(inp)[1:]

    xi = Input(shape=input_shape)
    x = separable_act_conv_bn(xi, input_shape[-1], ksize)

    model = Model(inputs=xi, outputs=x, name=name)

    return model(inp)


def build_regmap_block(inp, num_maps, name=None):
    input_shape = K.int_shape(inp)[1:]

    xi = Input(shape=input_shape)
    x = act_conv(xi, num_maps, (1, 1))

    model = Model(inputs=xi, outputs=x, name=name)

    return model(inp)


def build_fremap_block(inp, num_filters, name=None):
    input_shape = K.int_shape(inp)[1:]

    xi = Input(shape=input_shape)
    x = act_conv_bn(xi, num_filters, (1, 1))

    model = Model(inputs=xi, outputs=x, name=name)

    return model(inp)


def pose_regression_context(h, num_joints, sam_s_model,
        sam_c_model, jprob_c_model, agg_model, jprob_s_model):

    # Split heatmaps for specialized and contextual information
    hs = Lambda(lambda x: x[:,:,:,:num_joints])(h)
    hc = Lambda(lambda x: x[:,:,:,num_joints:])(h)

    # Soft-argmax and joint probability for each heatmap
    ps = sam_s_model(hs)
    pc = sam_c_model(hc)
    vc = jprob_c_model(hc)

    pose = agg_model([ps, pc, vc])
    visible = jprob_s_model(hs)

    return pose, visible, hs


def pose_regression(h, sam_s_model, jprob_s_model):

    pose = sam_s_model(h)
    visible = jprob_s_model(h)

    return pose, visible, h


def build_softargmax_2d(input_shape, name=None):

    if name is None:
        name_sm = None
    else:
        name_sm = name + '_softmax'

    inp = Input(shape=input_shape)
    x = act_channel_softmax(inp, name=name_sm)

    x_x = lin_interpolation_2d(x, dim=0)
    x_y = lin_interpolation_2d(x, dim=1)
    x = concatenate([x_x, x_y])

    model = Model(inputs=inp, outputs=x, name=name)
    model.trainable = False

    return model


def build_joints_probability(input_shape, name=None):

    num_rows, num_cols = input_shape[0:2]
    inp = Input(shape=input_shape)

    x = MaxPooling2D((num_rows, num_cols))(inp)
    x = Activation('sigmoid')(x)

    x = Lambda(lambda x: K.squeeze(x, axis=1))(x)
    x = Lambda(lambda x: K.squeeze(x, axis=1))(x)
    x = Lambda(lambda x: K.expand_dims(x, axis=-1))(x)

    model = Model(inputs=inp, outputs=x, name=name)

    return model


def build_context_aggregation(num_joints, num_context, alpha,
        num_frames=1, name=None):

    inp = Input(shape=(num_joints * num_context, 1))
    d = Dense(num_joints, use_bias=False)

    x = Lambda(lambda x: K.squeeze(x, axis=-1))(inp)
    x = d(x)
    x = Lambda(lambda x: K.expand_dims(x, axis=-1))(x)

    w = d.get_weights()
    w[0].fill(0)
    for j in range(num_joints):
        start = j*num_context
        w[0][j * num_context : (j + 1) * num_context, j] = 1.
    d.set_weights(w)
    d.trainable = False

    ctx_sum = Model(inputs=inp, outputs=x)
    ctx_sum.trainable = False
    if num_frames > 1:
        ctx_sum = TimeDistributed(ctx_sum,
                input_shape=(num_frames,) + K.int_shape(inp)[1:])

    # Define auxiliary layers.
    mul_alpha = Lambda(lambda x: alpha * x)
    mul_1alpha = Lambda(lambda x: (1 - alpha) * x)

    # This depends on TensorFlow because keras does not implement divide.
    tf_div = Lambda(lambda x: tf.divide(x[0], x[1]))

    if num_frames == 1:
        # Define inputs
        ys = Input(shape=(num_joints, 2))
        yc = Input(shape=(num_joints * num_context, 2))
        pc = Input(shape=(num_joints * num_context, 1))

        # Split contextual predictions in x and y and do computations separately
        xi = Lambda(lambda x: x[:,:, 0:1])(yc)
        yi = Lambda(lambda x: x[:,:, 1:2])(yc)
    else:
        ys = Input(shape=(num_frames, num_joints, 2))
        yc = Input(shape=(num_frames, num_joints * num_context, 2))
        pc = Input(shape=(num_frames, num_joints * num_context, 1))

        # Split contextual predictions in x and y and do computations separately
        xi = Lambda(lambda x: x[:,:,:, 0:1])(yc)
        yi = Lambda(lambda x: x[:,:,:, 1:2])(yc)

    pxi = multiply([xi, pc])
    pyi = multiply([yi, pc])

    pc_sum = ctx_sum(pc)
    pxi_sum = ctx_sum(pxi)
    pyi_sum = ctx_sum(pyi)
    pc_div = Lambda(lambda x: x / num_context)(pc_sum)
    pxi_div = tf_div([pxi_sum, pc_sum])
    pyi_div = tf_div([pyi_sum, pc_sum])
    yc_div = concatenate([pxi_div, pyi_div])

    ys_alpha = mul_alpha(ys)
    yc_div_1alpha = mul_1alpha(yc_div)

    y = add([ys_alpha, yc_div_1alpha])

    model = Model(inputs=[ys, yc, pc], outputs=y, name=name)
    model.trainable = False

    return model


def build(input_shape, num_joints,
        num_context_per_joint=2,
        alpha=0.8,
        num_blocks=8,
        ksize=(5, 5),
        export_heatmaps=False):

    inp = Input(shape=input_shape)
    outputs = []

    num_heatmaps = (num_context_per_joint + 1) * num_joints

    x = stem(inp)

    num_rows, num_cols, num_filters = K.int_shape(x)[1:]

    # Build the soft-argmax models (no parameters) for specialized and
    # contextual maps.
    sams_input_shape = (num_rows, num_cols, num_joints)
    sam_s_model = build_softargmax_2d(sams_input_shape, name='sSAM')
    jprob_s_model = build_joints_probability(sams_input_shape, name='sjProb')

    # Build the aggregation model (no parameters)
    if num_context_per_joint > 0:
        samc_input_shape = (num_rows, num_cols, num_heatmaps - num_joints)
        sam_c_model = build_softargmax_2d(samc_input_shape, name='cSAM')
        jprob_c_model = build_joints_probability(samc_input_shape,
                name='cjProb')
        agg_model = build_context_aggregation(num_joints,
                num_context_per_joint, alpha, name='Agg')

    for bidx in range(num_blocks):
        block_shape = K.int_shape(x)[1:]
        x = build_reception_block(x, name='rBlock%d' % (bidx + 1), ksize=ksize)

        ident_map = x
        x = build_sconv_block(x, name='SepConv%d' % (bidx + 1), ksize=ksize)
        h = build_regmap_block(x, num_heatmaps, name='RegMap%d' % (bidx + 1))

        if num_context_per_joint > 0:
            pose, visible, hm = pose_regression_context(h, num_joints,
                    sam_s_model, sam_c_model, jprob_c_model, agg_model,
                    jprob_s_model)
        else:
            pose, visible, hm = pose_regression(h, sam_s_model, jprob_s_model)

        outputs.append(pose)
        outputs.append(visible)
        if export_heatmaps:
            outputs.append(hm)

        if bidx < num_blocks - 1:
            h = build_fremap_block(h, block_shape[-1],
                    name='fReMap%d' % (bidx + 1))
            x = add([ident_map, x, h])

    model = Model(inputs=inp, outputs=outputs)

    return model

