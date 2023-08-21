import glob
import os
import cv2
import mmcv

def split_rgb(imagedir):
    rgb_data = glob.glob(os.path.join(imagedir, 'frame*.png'))
  
    rgb_list  = [os.path.basename(rgb) for rgb in rgb_data] 

    return rgb_list,

def get_frames(filename='test.mp4', output_path='./', interval=1):

    print("Spliting video to frames with an interval of {} ...".format(interval))
    video = mmcv.VideoReader(filename)

    # obtain basic information
    print(len(video))
    print(video.width, video.height, video.resolution, video.fps)

    img = video[0:-1:interval]
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for i in range(len(img)):
        name = os.path.join(output_path, 'frame%05d'%i + '.png')
        cv2.imwrite(name, img[i])
    print("Done with {} frames.".format(len(img)))

if __name__ == '__main__':
    get_frames(filename='/home/perple/Public/DROID-SLAM/data/zn_office/zn_office.mp4', output_path='/home/perple/Public/DROID-SLAM/data/zn_office')