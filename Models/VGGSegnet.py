




from keras.models import *
from keras.layers import *

import tensorflow as tf

import os
file_path = os.path.dirname( os.path.abspath(__file__) )
VGG_Weights_path = file_path+"/../data/vgg16_weights_th_dim_ordering_th_kernels.h5"

import keras.backend as K

def sobel_layer(image):
    sobel_x = tf.constant([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], tf.float32)
    sobel_x_filter = tf.reshape(sobel_x, [1, 1, 3, 3])
    sobel_y_filter = tf.transpose(sobel_x_filter, [1, 0, 2, 3])

    # Shape = height x width.
    #image = tf.placeholder(tf.float32, shape=[None, None])

    # Shape = 1 x height x width x 1.
    #image_resized = tf.expand_dims(tf.expand_dims(image, 0), 3)

    #filtered_x = tf.nn.conv2d(image, sobel_x_filter,
                              #strides=[1, 1, 1, 1], padding='SAME', data_format='NCHW')
    #filtered_y = tf.nn.conv2d(image, sobel_y_filter,
                              #strides=[1, 1, 1, 1], padding='SAME', data_format='NCHW')
    #filtered_x = Conv2D(image, strides=(1, 1), padding='same', data_format='channels_first', name='sobel_x')
    #filtered_y = Conv2D(image, strides=(1, 1), padding='same', data_format='channels_first', name='sobel_y')

    sobelFilter = K.variable([[[[1.,  1.]], [[0.,  2.]],[[-1.,  1.]]],
                              [[[2.,  0.]], [[0.,  0.]],[[-2.,  0.]]],
                              [[[1., -1.]], [[0., -2.]],[[-1., -1.]]]])

    out = sobelFilter * K.reshape(image[0,:,0,0],(1,1,-1,1))
    return out #add([filtered_x, filtered_y])

def VGGSegnet( n_classes ,  input_height=416, input_width=608 , vgg_level=3, use_pooling=True, use_fc_layers=False):

    img_input = Input(shape=(3,input_height,input_width))

    #x = Conv2D(3, (3, 3), activation='relu', padding='same', name='sobel1_conv1', data_format='channels_first' )(img_input)

    #x = multiply([x, img_input])
    x = img_input

    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', data_format='channels_first' )(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', data_format='channels_first' )(x)
    if use_pooling:
        x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block1_pool', data_format='channels_first' )(x)
    f1 = x
    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', data_format='channels_first' )(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', data_format='channels_first' )(x)
    if use_pooling:
        x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block2_pool', data_format='channels_first' )(x)
    f2 = x

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', data_format='channels_first' )(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', data_format='channels_first' )(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', data_format='channels_first' )(x)
    if use_pooling:
        x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block3_pool', data_format='channels_first' )(x)
    f3 = x

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', data_format='channels_first' )(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', data_format='channels_first' )(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', data_format='channels_first' )(x)
    if use_pooling and False:
        x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block4_pool', data_format='channels_first' )(x)
    f4 = x

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', data_format='channels_first' )(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', data_format='channels_first' )(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', data_format='channels_first' )(x)
    if use_pooling and False:
        x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block5_pool', data_format='channels_first' )(x)
    f5 = x

    if use_fc_layers:
        x = Flatten(name='flatten')(x)
        x = Dense(4096, activation='relu', name='fc1')(x)
        x = Dense(4096, activation='relu', name='fc2')(x)
        x = Dense( 1000 , activation='softmax', name='predictions')(x)

        vgg  = Model(  img_input , x  )

        if os.path.exists(VGG_Weights_path):
            vgg.load_weights(VGG_Weights_path)
    else:
        x = Conv2D(4096, (1, 1), activation='relu', padding='same', name='fcn1', data_format='channels_first' )(x)
        x = Conv2D(4096, (1, 1), activation='relu', padding='same', name='fcn2', data_format='channels_first' )(x)
        x = Conv2D(1000, (1, 1), activation='softmax', padding='same', name='fcn3', data_format='channels_first' )(x)

    levels = [f1 , f2 , f3 , f4 , f5 ]

    o = levels[ vgg_level ]

    o = x
    
    #o = ( ZeroPadding2D( (1,1) , data_format='channels_first' ))(o)
    o = ( Conv2DTranspose(512, (3, 3), padding='same', data_format='channels_first'))(o)
    o = ( BatchNormalization())(o)

    o = add([o, f4])
    if use_pooling and False:
        o = ( UpSampling2D( (2,2), data_format='channels_first'))(o)
    #o = ( ZeroPadding2D( (1,1), data_format='channels_first'))(o)
    o = ( Conv2DTranspose( 256, (3, 3), strides=(1,1), padding='same', data_format='channels_first'))(o)
    o = ( BatchNormalization())(o)
 
    o = add([o, f3])
    if use_pooling:
        o = ( UpSampling2D((2,2)  , data_format='channels_first' ) )(o)
    #o = ( ZeroPadding2D((1,1) , data_format='channels_first' ))(o)
    o = ( Conv2DTranspose( 128 , (3, 3), strides=(1,1), padding='same' , data_format='channels_first' ))(o)
    o = ( BatchNormalization())(o)

    o = add([o, f2])
    if use_pooling:
        o = ( UpSampling2D((2,2)  , data_format='channels_first' ))(o)
    #o = ( ZeroPadding2D((1,1)  , data_format='channels_first' ))(o)
    o = ( Conv2DTranspose( 64 , (3, 3), strides=(1,1), padding='same'  , data_format='channels_first' ))(o)
    o = ( BatchNormalization())(o)

    o = add([o, f1])
    if use_pooling:
        o = ( UpSampling2D((2,2)  , data_format='channels_first' ))(o)
    o =  Conv2DTranspose( n_classes , (3, 3) , padding='same', activation='relu', data_format='channels_first' )( o )

    o_shape = Model(img_input , o ).output_shape
    outputHeight = o_shape[2]
    outputWidth = o_shape[3]

    o = (Reshape(( -1  , outputHeight*outputWidth)))(o)
    o = (Permute((2, 1)))(o)
    o = (Activation('softmax'))(o)
    model = Model( img_input , o )
    model.outputWidth = outputWidth
    model.outputHeight = outputHeight
    '''
    o = (Reshape(( -1  , outputHeight*outputWidth )))(o)
    o = (Permute((2, 1)))(o)
    model = Model( img_input , o )
    model.outputWidth = outputWidth
    model.outputHeight = outputHeight
    '''

    return model


if __name__ == '__main__':
    m = VGGSegnet( 101 )
    from keras.utils import plot_model
    plot_model( m , show_shapes=True , to_file='model.png')

