import tensorflow as tf
from tensorflow.keras.layers import (
    Conv2D,
    Conv2DTranspose,
    MaxPooling2D,
    BatchNormalization,
    ReLU,
    Concatenate
)


def conv_block(x, filters):

    x = Conv2D(filters, kernel_size=3, padding="same", use_bias=True)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2D(filters, kernel_size=3, padding="same", use_bias=True)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    return x



def match_size(x, skip):

    x_height = tf.shape(x)[1]
    x_width = tf.shape(x)[2]

    skip_height = tf.shape(skip)[1]
    skip_width = tf.shape(skip)[2]


    min_height = tf.minimum(x_height, skip_height)
    min_width = tf.minimum(x_width, skip_width)

    x = x[:, :min_height, :min_width, :]
    skip = skip[:, :min_height, :min_width, :]

    return x, skip


def upsample_and_fuse(x, skip, filters):

    x = Conv2DTranspose(filters, kernel_size=2, strides=2, padding="same")(x)

    x, skip = match_size(x, skip)

    x = Concatenate()([x, skip])

    x = conv_block(x, filters)

    return x


def build_unet():

    inputs = tf.keras.Input(shape=(None, None, 1))


    e1 = conv_block(inputs, 16)
    p1 = MaxPooling2D(pool_size=2)(e1)

    e2 = conv_block(p1, 32)
    p2 = MaxPooling2D(pool_size=2)(e2)

    e3 = conv_block(p2, 64)
    p3 = MaxPooling2D(pool_size=2)(e3)
    
    e4 = conv_block(p3, 128)
    p4 = MaxPooling2D(pool_size=2)(e4)



    b = conv_block(p4, 256)



    d4 = upsample_and_fuse(b, e4, 128)

    d3 = upsample_and_fuse(d4, e3, 64)

    d2 = upsample_and_fuse(d3, e2, 32)

    d1 = upsample_and_fuse(d2, e1, 16)


    outputs = Conv2D(1, kernel_size=1, padding="same")(d1)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model