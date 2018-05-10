import glob
import numpy as np
import cv2
import random
import argparse

'''
0   None
1   Buildings
2   Fences
3   Other
4   Pedestrians
5   Poles
6   RoadLines
7   Roads
8   Sidewalks
9   Vegetation
10  Vehicles
11  Walls
12  TrafficSigns
'''
colors = {0: (0,0,0), 
          1: (150,150,150),
          2: (80,80,80),
          3: (50,50,50),
          4: (205,205,0),
          5: (0,200,200),
          6: (255,255,255),
          7: (255,255,255),
          8: (255,0,0),
          9: (0,255,0),
         10: (0,0,250),
         11: (255,0,255),
         12: (100,100,100)
         }


def visualizeImage(img, seg, n_classes, gt=None, render=True):
    # remove hood part of image
    img = img[:-102,:,:]
    seg = seg[:-102,:,:]

    seg_img = np.zeros_like( seg )

    for c in range(n_classes):
        if c == 7 or c == 6 or c == 10:
            seg_img[:,:,0] += ((seg[:,:,2] == c )*( colors[c][0] )).astype('uint8')
            seg_img[:,:,1] += ((seg[:,:,2] == c )*( colors[c][1] )).astype('uint8')
            seg_img[:,:,2] += ((seg[:,:,2] == c )*( colors[c][2] )).astype('uint8')

    vis1 = np.concatenate((img, seg_img), axis=1)

    seg_img2 = cv2.addWeighted(seg_img,0.7, img, 0.3, 0, dtype=cv2.CV_8UC1)
    if gt is not None:
        vis2 = np.concatenate((seg_img2, gt), axis=1)
    else:
        vis2 = np.concatenate((seg_img2, seg_img2), axis=1)

    if render:
        vis2 = np.concatenate((vis1, vis2), axis=0)
        cv2.imshow("Annotated image", vis2)

        c = cv2.waitKey() & 0x7F
        if c == 27 or c == ord('q'):
            exit()
    return seg_img2


def visualizeImages( images_path , segs_path ,  n_classes ):
    assert images_path[-1] == '/'
    assert segs_path[-1] == '/'

    images = glob.glob( images_path + "*.jpg"  ) + glob.glob( images_path + "*.png"  ) +  glob.glob( images_path + "*.jpeg"  )
    images.sort()
    segmentations  = glob.glob( segs_path + "*.jpg"  ) + glob.glob( segs_path + "*.png"  ) +  glob.glob( segs_path + "*.jpeg"  )
    segmentations.sort()

    assert len( images ) == len(segmentations)

    for im_fn , seg_fn in zip(images,segmentations):
        assert(  im_fn.split('/')[-1] ==  seg_fn.split('/')[-1] )

        img = cv2.imread( im_fn )
        seg = cv2.imread( seg_fn )
        print np.unique( seg )

        visualizeImage(img, seg, n_classes)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type = str  )
    parser.add_argument("--annotations", type = str  )
    parser.add_argument("--n_classes", type=int )
    args = parser.parse_args()

    visualizeImages(args.images ,  args.annotations  ,  args.n_classes   ) 
