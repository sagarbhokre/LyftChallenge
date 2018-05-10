import argparse
import Models , LoadBatches
import glob
import itertools
import cv2
import numpy as np
import tensorflow as tf
from keras import backend as K

def filter_mult(image):
    sobel_x = (1.0/8)*tf.constant([[1, 0, -1], [2, 0, -2], [1, 0, -1]], tf.float32)
    sobel_x_filter = tf.reshape(sobel_x, [3, 3, 1, 1])
    sobel_y_filter = tf.transpose(sobel_x_filter, [1, 0, 2, 3])

    image_resized = tf.reshape(image, (-1,600,800,12))

    filtered_x = []
    filtered_y = []
    #im_planes = tf.split(image_resized, num_or_size_splits=12, axis=3, name='split')
    im_planes = tf.unstack(image_resized, num=12, axis=3)

    for i in range(len(im_planes)):
        image_resized = tf.expand_dims(im_planes[0], 3)

        x = tf.nn.conv2d(image_resized, sobel_x_filter,
                         strides=[1, 1, 1, 1], padding='SAME')
        y = tf.nn.conv2d(image_resized, sobel_y_filter,
                         strides=[1, 1, 1, 1], padding='SAME')

        filtered_x.append(x)
        filtered_y.append(y)

    filtered_x = tf.squeeze(tf.stack(filtered_x, axis=3), axis=4)
    filtered_y = tf.squeeze(tf.stack(filtered_y, axis=3), axis=4)

    result = filtered_x + filtered_y
    result = tf.reshape(result, (-1, 600*800, 12))
    return result

def mean_iou_loss(y_true, y_pred):
    inter=tf.reduce_sum(tf.multiply(y_pred,y_true))
    union=tf.reduce_sum(tf.subtract(tf.add(y_pred,y_true),tf.multiply(y_pred,y_true)))
    loss=tf.subtract(tf.constant(1.0, dtype=tf.float32),tf.div(inter,union))
    return loss

def mean_iou(y_true, y_pred):
    #y_pred = filter_mult(y_pred)
    y_pred = tf.to_int32(y_pred > 0.5)
    score, up_opt = tf.metrics.mean_iou(y_true, y_pred, 12)#, weights = tf.constant([0.01, 0.99]))
    K.get_session().run(tf.local_variables_initializer())
    with tf.control_dependencies([up_opt]):
        score = tf.identity(score)
    return score

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_weights_path", type = str )
    parser.add_argument("--train_images", type = str )
    parser.add_argument("--train_annotations", type = str )
    parser.add_argument("--n_classes", type=int )
    parser.add_argument("--input_height", type=int , default = 600 )
    parser.add_argument("--input_width", type=int , default = 800 )

    parser.add_argument('--validate',action='store_false')
    parser.add_argument("--val_images", type = str , default = "")
    parser.add_argument("--val_annotations", type = str , default = "")

    parser.add_argument("--epochs", type = int, default = 5 )
    parser.add_argument("--batch_size", type = int, default = 2 )
    parser.add_argument("--val_batch_size", type = int, default = 1 )
    parser.add_argument("--load_weights", type = str , default = "")

    parser.add_argument("--model_name", type = str , default = "")
    parser.add_argument("--optimizer_name", type = str , default = "adadelta")

    args = parser.parse_args()

    train_images_path = args.train_images
    train_segs_path = args.train_annotations
    train_batch_size = args.batch_size
    n_classes = args.n_classes
    input_height = args.input_height
    input_width = args.input_width
    validate = args.validate
    save_weights_path = args.save_weights_path
    epochs = args.epochs
    load_weights = args.load_weights

    optimizer_name = args.optimizer_name
    model_name = args.model_name

    if validate:
        val_images_path = args.val_images
        val_segs_path = args.val_annotations
        val_batch_size = args.val_batch_size

    modelFns = { 'vgg_segnet':Models.VGGSegnet.VGGSegnet , 'vgg_unet':Models.VGGUnet.VGGUnet , 'vgg_unet2':Models.VGGUnet.VGGUnet2 , 'fcn8':Models.FCN8.FCN8 , 'fcn32':Models.FCN32.FCN32   }
    modelFN = modelFns[ model_name ]

    m = modelFN( n_classes , input_height=input_height, input_width=input_width   )
    m.compile(loss=mean_iou_loss, #'categorical_crossentropy',
              optimizer= optimizer_name ,
              metrics=['accuracy', mean_iou])

    if load_weights  != "":
        m.load_weights(load_weights)

    print "Model output shape" ,  m.output_shape

    output_height = m.outputHeight
    output_width = m.outputWidth

    G  = LoadBatches.imageSegmentationGenerator( train_images_path , train_segs_path ,  train_batch_size,  n_classes , input_height , input_width , output_height , output_width   )

    '''
    X,Y = G.next()
    Y = np.reshape(Y, (output_height, output_width, n_classes))
    print("Y shape: ", Y.shape, " max Y: ", np.max(Y))
    for i in range(n_classes):
        print("Plane %d sum: %d"%(i, np.sum(Y[:,:,i])))
    print("All Plane sum: ", np.sum(Y))
    '''

    if validate:
        G2  = LoadBatches.imageSegmentationGenerator( val_images_path , val_segs_path ,  val_batch_size,  n_classes , input_height , input_width , output_height , output_width   )

    for ep in range( epochs ):
        print("Epoch: %d"%(ep))
        if not validate:
            m.fit_generator( G , 512  , epochs=1 )
        else:
            m.fit_generator( G , 512  , validation_data=G2 , validation_steps=200 ,  epochs=1 )

        m.save_weights( save_weights_path + "." + str( ep ) )
        m.save( save_weights_path + ".model." + str( ep ) )
