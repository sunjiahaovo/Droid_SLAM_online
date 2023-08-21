import glob
import os
import cv2

def split_rgbd(imagedir):
    rgb_data = glob.glob(os.path.join(imagedir, 'frame*.png'))
    depth_data = glob.glob(os.path.join(imagedir, 'depth*.png'))

    rgb_list  = [os.path.basename(rgb) for rgb in rgb_data] 
    depth_list = [os.path.basename(depth) for depth in depth_data]

    return rgb_list, depth_list

def rename(imagedir):
    rgb_data = glob.glob(os.path.join(imagedir, '*.png'))
    depth_data = glob.glob(os.path.join(imagedir, '*_depth.png'))

    rgb_list  = [os.path.basename(rgb) for rgb in rgb_data] 
    depth_list = [os.path.basename(depth) for depth in depth_data]  

    for rgb, depth in zip(rgb_list, depth_list):
        rgb_num = int(rgb.split('_')[0])   
        depth_num = int(depth.split('_')[0]) 
        rgb_new_name = f'frame{rgb_num:05d}.png'
        depth_new_name = f'depth{depth_num:05d}.png'
        os.rename(os.path.join(imagedir, rgb), os.path.join(imagedir, rgb_new_name))
        os.rename(os.path.join(imagedir, depth), os.path.join(imagedir, depth_new_name))

if __name__ == '__main__':
    rename('/remote-home/ums_sunjiahao/droid-slam/data/bear_room_ipad')
    # split_rgbd('/remote-home/ums_sunjiahao/droid-slam/data/bear_room_realsense_online_ellipse')