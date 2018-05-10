import argparse
import Models , LoadBatches
from keras.models import load_model
import glob
import cv2
import numpy as np
import random
import os
from visualizeDataset import visualizeImage

def sortKeyFunc(s):
    return int(os.path.basename(s)[:-4])

parser = argparse.ArgumentParser()
parser.add_argument("--save_weights_path", type = str  )
parser.add_argument("--epoch_number", type = int, default = 5 )
parser.add_argument("--test_images", type = str , default = "")
parser.add_argument("--output_path", type = str , default = "")
parser.add_argument("--input_height", type=int , default = 600  )
parser.add_argument("--input_width", type=int , default = 800 )
parser.add_argument("--model_name", type = str , default = "")
parser.add_argument("--n_classes", type=int )

args = parser.parse_args()

n_classes = args.n_classes
model_name = args.model_name
images_path = args.test_images
input_width =  args.input_width
input_height = args.input_height
epoch_number = args.epoch_number

modelFns = { 'vgg_segnet':Models.VGGSegnet.VGGSegnet , 'vgg_unet':Models.VGGUnet.VGGUnet , 'vgg_unet2':Models.VGGUnet.VGGUnet2 , 'fcn8':Models.FCN8.FCN8 , 'fcn32':Models.FCN32.FCN32   }
modelFN = modelFns[ model_name ]

m = modelFN( n_classes , input_height=input_height, input_width=input_width   )
m.load_weights(  args.save_weights_path + "." + str(  epoch_number )  )
m.compile(loss='categorical_crossentropy',
      optimizer= 'adadelta' ,
      metrics=['accuracy'])


output_height = m.outputHeight
output_width = m.outputWidth

images = glob.glob( images_path + "*.jpg"  ) + glob.glob( images_path + "*.png"  ) +  glob.glob( images_path + "*.jpeg"  )
#images.sort()#key=os.path.getmtime)
images.sort(key=sortKeyFunc)

colors = [  ( random.randint(0,255),random.randint(0,255),random.randint(0,255)   ) for _ in range(n_classes)  ]

#fourcc = cv2.VideoWriter_fourcc(*'XVID')
fourcc = cv2.cv.CV_FOURCC(*'XVID')
w,h = args.input_width, args.input_height

#videoout = None
videoout = cv2.VideoWriter('output.avi',fourcc, 24.0, (w, h))

for idx, imgName in enumerate(images):
    if idx % 100 == 0:
        print ("%d/%d"%(idx, len(images)))

    in_img = cv2.imread(imgName)
    outName = imgName.replace( images_path ,  args.output_path )

    X = LoadBatches.getImageArr(imgName , args.input_width  , args.input_height  )

    pr = m.predict( np.array([X]) )[0]
    pr = pr.reshape(( output_height ,  output_width , n_classes ) ).argmax( axis=2 )

    seg_img = np.zeros( ( output_height , output_width , 3 ) )
    seg_arr = np.zeros( ( output_height , output_width , 3 ) , dtype='uint8')

    seg_arr[:,:,2] = (pr).astype('uint8')

    for c in range(n_classes):
        seg_img[:,:,0] += ((pr[:,:] == c )*( colors[c][0] )).astype('uint8')
        seg_img[:,:,1] += ((pr[:,:] == c )*( colors[c][1] )).astype('uint8')
        seg_img[:,:,2] += ((pr[:,:] == c )*( colors[c][2] )).astype('uint8')

    in_img = cv2.resize(in_img, (args.input_width, args.input_height))
    seg_img = cv2.resize(seg_img  , (input_width , input_height ))

    seg_img = cv2.addWeighted(seg_img,0.7,in_img,0.3,0, dtype=cv2.CV_8UC1)

    if videoout is not None:
        seg_img = visualizeImage(in_img, seg_arr, n_classes, render=False)
        seg_img = cv2.resize(seg_img  , (w , h ))
        videoout.write(seg_img)
    else:
        visualizeImage(in_img, seg_arr, n_classes)

if videoout is not None:
    videoout.release()
exit()
