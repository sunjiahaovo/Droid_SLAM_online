import glob
import os
import cv2

def split_rgbd(imagedir):
    rgb_data = glob.glob(os.path.join(imagedir, 'frame*.jpg'))
    depth_data = glob.glob(os.path.join(imagedir, 'depth*.png'))

    rgb_list  = [os.path.basename(rgb) for rgb in rgb_data] 
    depth_list = [os.path.basename(depth) for depth in depth_data]

    return rgb_list, depth_list



if __name__ == '__main__':
    rgb_list, depth_list = split_rgbd('../dataset/Replica/room0/results')
    print('split successfully')