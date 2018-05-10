import sys, skvideo.io, json, base64
import numpy as np
from PIL import Image
from io import BytesIO, StringIO
import cv2

# TODO: Delete this import
from visualizeDataset import visualizeImage

import Models, LoadBatches
from keras.models import load_model
import numpy as np

n_classes = 12
input_width = 800
input_height = 600
visualize = False

model_path = 'weights/ex1.model.35'

def load_seg_model():
    m = Models.VGGSegnet.VGGSegnet(n_classes, input_height=input_height, input_width=input_width)
    m.load_weights(model_path)
    m.compile(loss='categorical_crossentropy',
              optimizer= 'adadelta' ,
              metrics=['accuracy'])

    output_height = m.outputHeight
    output_width = m.outputWidth
    return m, output_width, output_height

def preprocess_img(img, ordering='channels_first'):
    #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = img.astype(np.float32)
    #img[:,:,0] -= 103.939
    #img[:,:,1] -= 116.779
    #img[:,:,2] -= 123.68
    img = img/255.0
    if ordering == 'channels_first':
        img = np.rollaxis(img, 2, 0)
    return img

if __name__ == '__main__':
    file = sys.argv[-1]

    if file == 'demo.py':
      print ("Error loading video")
      quit

    # Define encoder function
    def encode(array):
        pil_img = Image.fromarray(array)
        buff = BytesIO()
        pil_img.save(buff, format="PNG")
        return base64.b64encode(buff.getvalue()).decode("utf-8")

    video = skvideo.io.vread(file)

    m, output_width, output_height = load_seg_model()

    answer_key = {}

    # Frame numbering starts at 1
    frame = 1

    for rgb_frame in video:

        X = preprocess_img(rgb_frame)

        pr = m.predict( np.array([X]) )[0]
        pr = pr.reshape((output_height, output_width, n_classes)).argmax(axis=2)

        seg_arr = np.zeros((output_height, output_width, 3) , dtype='uint8')

        seg_arr[:,:,2] = (pr).astype('uint8')

        #import pdb; pdb.set_trace()
        binary_car_result = np.where((pr==10),1,0).astype('uint8')

        binary_road_result = np.where(((pr == 7) | (pr == 6)),1,0).astype('uint8')
        binary_road_result1 = np.where(((pr == 7)),1,0).astype('uint8')
        binary_road_result2 = np.where(((pr == 6)),1,0).astype('uint8')

        answer_key[frame] = [encode(binary_car_result), encode(binary_road_result)]

        if visualize:
            seg_img = visualizeImage(rgb_frame, seg_arr, n_classes, render=True)

        # Increment frame
        frame+=1

    # Print output in proper json format
    print (json.dumps(answer_key))
