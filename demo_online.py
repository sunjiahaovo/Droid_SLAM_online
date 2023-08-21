import sys
sys.path.append('droid_slam')

from tqdm import tqdm
import numpy as np
import torch
import lietorch
import cv2
import os
import glob 
import time
import argparse

from preprocess.realsense_utils import split_rgbd
# from preprocess.replica_utils import split_rgbd
from preprocess.moblie_utils import split_rgb

from torch.multiprocessing import Process
from droid import Droid

import torch.nn.functional as F


def show_image(image):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey(1)

# read rgb online
def rgb_online(imagedir, calib, stride, start, update, t, image_list):
    calib = np.loadtxt(calib, delimiter=" ")
    fx, fy, cx, cy = calib[:4]
    
    K = np.eye(3)
    K[0,0] = fx
    K[0,2] = cx
    K[1,1] = fy
    K[1,2] = cy
    

    while(start):
        if os.listdir(imagedir):
            image_list = sorted(os.listdir(imagedir))[::stride]
            start = 0
        else:
            continue
    
    
    tracking = 1

    imfile = image_list[t]
    # for t, imfile in enumerate(image_list):
    while(tracking):
        update_list = sorted(os.listdir(imagedir))[::stride]
        waiting_image = 0
        
        if len(image_list) < len(update_list):
            image_list = update_list
            waiting_image = 0
        if len(image_list) == len(update_list) and t+1 == len(image_list):
            waiting_image = 1
            image = cv2.imread(os.path.join(imagedir, imfile))
            if len(calib) > 4:
                image = cv2.undistort(image, K, calib[4:])

            h0, w0, _ = image.shape
            h1 = int(h0 * np.sqrt((384 * 512) / (h0 * w0)))
            w1 = int(w0 * np.sqrt((384 * 512) / (h0 * w0)))

            image = cv2.resize(image, (w1, h1))
            image = image[:h1-h1%8, :w1-w1%8]
            image = torch.as_tensor(image).permute(2, 0, 1)

            intrinsics = torch.as_tensor([fx, fy, cx, cy])
            intrinsics[0::2] *= (w1 / w0)
            intrinsics[1::2] *= (h1 / h0)

            yield t, image[None], intrinsics, waiting_image, image_list
            while(update):
                update_list = sorted(os.listdir(imagedir))[::stride]
                if len(image_list) < len(update_list):
                    image_list = update_list
                    update = 0
                else:
                    continue
        
        
        image = cv2.imread(os.path.join(imagedir, imfile))
        # image = image[::-1, ::-1, 0]
        if len(calib) > 4:
            image = cv2.undistort(image, K, calib[4:])

        h0, w0, _ = image.shape
        h1 = int(h0 * np.sqrt((384 * 512) / (h0 * w0)))
        w1 = int(w0 * np.sqrt((384 * 512) / (h0 * w0)))

        image = cv2.resize(image, (w1, h1))
        image = image[:h1-h1%8, :w1-w1%8]
        image = torch.as_tensor(image).permute(2, 0, 1)

        intrinsics = torch.as_tensor([fx, fy, cx, cy])
        intrinsics[0::2] *= (w1 / w0)
        intrinsics[1::2] *= (h1 / h0)

        yield t, image[None], intrinsics, waiting_image, image_list
        t += 1
        image_list = sorted(os.listdir(imagedir))[::stride]
        imfile = image_list[t]
     
def rgb_stream(imagedir, calib, stride):
    """ image generator """

    calib = np.loadtxt(calib, delimiter=" ")
    fx, fy, cx, cy = calib[:4]

    K = np.eye(3)
    K[0,0] = fx
    K[0,2] = cx
    K[1,1] = fy
    K[1,2] = cy

    image_list = sorted(os.listdir(imagedir))[::stride]
    # image_list = split_rgb(imagedir)[::stride]

    for t, imfile in enumerate(image_list):
        image = cv2.imread(os.path.join(imagedir, imfile))
        # image = image[::-1, ::-1, 0]
        if len(calib) > 4:
            image = cv2.undistort(image, K, calib[4:])

        h0, w0, _ = image.shape
        h1 = int(h0 * np.sqrt((384 * 512) / (h0 * w0)))
        w1 = int(w0 * np.sqrt((384 * 512) / (h0 * w0)))

        image = cv2.resize(image, (w1, h1))
        image = image[:h1-h1%8, :w1-w1%8]
        image = torch.as_tensor(image).permute(2, 0, 1)

        intrinsics = torch.as_tensor([fx, fy, cx, cy])
        intrinsics[0::2] *= (w1 / w0)
        intrinsics[1::2] *= (h1 / h0)

        yield t, image[None], intrinsics

# read rgbd_online
def rgbd_online(imagedir, calib, stride, start, update, t, image_list):
    calib = np.loadtxt(calib, delimiter=" ")
    fx, fy, cx, cy = calib[:4]
    
    K = np.eye(3)
    K[0,0] = fx
    K[0,2] = cx
    K[1,1] = fy
    K[1,2] = cy
    
    image_list, depth_list = split_rgbd(imagedir)
    image_list = sorted(image_list)[::stride]
    depth_list = sorted(depth_list)[::stride]

    while(start):
        if os.listdir(imagedir):
            image_list, depth_list = split_rgbd(imagedir)
            image_list = sorted(image_list)[::stride]
            depth_list = sorted(depth_list)[::stride]
            start = 0
        else:
            continue
    
    
    tracking = 1

    imfile = image_list[t]
    dpfile = image_list[t]
    # for t, imfile in enumerate(image_list):
    while(tracking):
        update_list, update_depth_list = split_rgbd(imagedir)
        update_list = sorted(update_list)[::stride]
        update_depth_list = sorted(update_depth_list)[::stride]
        waiting_image = 0
        
        if len(image_list) < len(update_list):
            image_list = update_list
            depth_list = update_depth_list
            waiting_image = 0
        if len(image_list) == len(update_list) and t+1 == len(image_list):
            waiting_image = 1
            image = cv2.imread(os.path.join(imagedir, imfile))
            depth = cv2.imread(os.path.join(imagedir, dpfile), cv2.IMREAD_ANYDEPTH)
            
            if len(calib) > 4:
                image = cv2.undistort(image, K, calib[4:])

            h0, w0, _ = image.shape
            h1 = int(h0 * np.sqrt((384 * 512) / (h0 * w0)))
            w1 = int(w0 * np.sqrt((384 * 512) / (h0 * w0)))

            image = cv2.resize(image, (w1, h1))
            image = image[:h1-h1%8, :w1-w1%8]
            image = torch.as_tensor(image).permute(2, 0, 1)
            
            depth = depth.astype(np.float32)
            depth = torch.as_tensor(depth)
            # for scale 
            depth /= 1000
            depth = F.interpolate(depth[None,None], (h1, w1)).squeeze()
            depth = depth[:h1-h1%8, :w1-w1%8]

            intrinsics = torch.as_tensor([fx, fy, cx, cy])
            intrinsics[0::2] *= (w1 / w0)
            intrinsics[1::2] *= (h1 / h0)

            yield t, image[None], depth, intrinsics, waiting_image, image_list
            while(update):
                update_list, update_depth_list = split_rgbd(imagedir)
                update_list = sorted(update_list)[::stride]
                update_depth_list = sorted(update_depth_list)[::stride]
                if len(image_list) < len(update_list):
                    image_list = update_list
                    depth_list = update_depth_list
                    update = 0
                else:
                    continue
        
        
        image = cv2.imread(os.path.join(imagedir, imfile))
        depth = cv2.imread(os.path.join(imagedir, dpfile), cv2.IMREAD_ANYDEPTH)
        # image = image[::-1, ::-1, 0]
        if len(calib) > 4:
            image = cv2.undistort(image, K, calib[4:])

        h0, w0, _ = image.shape
        h1 = int(h0 * np.sqrt((384 * 512) / (h0 * w0)))
        w1 = int(w0 * np.sqrt((384 * 512) / (h0 * w0)))

        image = cv2.resize(image, (w1, h1))
        image = image[:h1-h1%8, :w1-w1%8]
        image = torch.as_tensor(image).permute(2, 0, 1)
        
        depth = depth.astype(np.float32)
        depth = torch.as_tensor(depth)
        # for scale 
        depth /= 1000
        depth = F.interpolate(depth[None,None], (h1, w1)).squeeze()
        depth = depth[:h1-h1%8, :w1-w1%8]

        intrinsics = torch.as_tensor([fx, fy, cx, cy])
        intrinsics[0::2] *= (w1 / w0)
        intrinsics[1::2] *= (h1 / h0)

        yield t, image[None], depth, intrinsics, waiting_image, image_list
        t += 1
        image_list, depth_list = split_rgbd(imagedir)
        image_list = sorted(image_list)[::stride]
        depth_list = sorted(depth_list)[::stride]
        imfile = image_list[t]
        dpfile = depth_list[t]

# add the depth input 
def rgbd_stream(imagedir, calib, stride):
    print(imagedir)
    calib = np.loadtxt(calib, delimiter=" ")
    fx, fy, cx, cy = calib[:4]

    K = np.eye(3)
    K[0, 0] = fx
    K[1, 1] = fy
    K[0, 2] = cx
    K[1, 2] = cy

    image_list, depth_list = split_rgbd(imagedir)
    image_list = sorted(image_list)[::stride]
    depth_list = sorted(depth_list)[::stride]

    for t, (imfile, dpfile) in enumerate(zip(image_list, depth_list)):
        rgb = cv2.imread(os.path.join(imagedir, imfile))
        depth = cv2.imread(os.path.join(imagedir, dpfile), cv2.IMREAD_ANYDEPTH)

        if len(calib) > 4 :
            rgb = cv2.undistort(rgb, K, calib[4:])
        
        
        h0, w0, _ = rgb.shape
        h1 = int(h0 * np.sqrt((384 * 512) / (h0 * w0)))
        w1 = int(w0 * np.sqrt((384 * 512) / (h0 * w0)))   

        rgb = cv2.resize(rgb, (w1, h1))
        rgb = rgb[:h1-h1%8, :w1-w1%8]
        rgb = torch.as_tensor(rgb).permute(2, 0, 1)

        depth = depth.astype(np.float32)
        depth = torch.as_tensor(depth)
        # for scale 
        depth /= 1000
        depth = F.interpolate(depth[None,None], (h1, w1)).squeeze()
        depth = depth[:h1-h1%8, :w1-w1%8]

        intrinsics = torch.as_tensor([fx, fy, cx, cy])
        intrinsics[0::2] *= (w1 / w0)
        intrinsics[1::2] *= (h1 / h0)

        yield t, rgb[None], intrinsics
        # yield t, rgb[None], depth, intrinsics

def save_initialization_reconstruction(droid, reconstruction_path):

    from pathlib import Path
    import random
    import string

    t = droid.video.counter.value-7
    tstamps = droid.video.tstamp[:t].cpu().numpy()
    images = droid.video.images[:t].cpu().numpy()
    disps_up = droid.video.disps_up[:t].cpu().numpy()
    poses = droid.video.poses[:t].cpu().numpy()
    intrinsics = droid.video.intrinsics[:t].cpu().numpy()
    disps = droid.video.disps[:t].cpu().numpy()

    timestamp = f"pic{t:03d}"
    Path("reconstructions/{}".format(reconstruction_path+timestamp)).mkdir(parents=True, exist_ok=True)
    np.save("reconstructions/{}/tstamps.npy".format(reconstruction_path+timestamp), tstamps)
    np.save("reconstructions/{}/images.npy".format(reconstruction_path+timestamp), images)
    np.save("reconstructions/{}/disps_up.npy".format(reconstruction_path+timestamp), disps_up)
    np.save("reconstructions/{}/disps.npy".format(reconstruction_path+timestamp), disps)
    np.save("reconstructions/{}/poses.npy".format(reconstruction_path+timestamp), poses)
    np.save("reconstructions/{}/intrinsics.npy".format(reconstruction_path+timestamp), intrinsics)

def save_reconstruction(droid, reconstruction_path):

    from pathlib import Path
    import random
    import string

    t = droid.video.counter.value-7
    tstamps = droid.video.tstamp[t-1:t].cpu().numpy()
    images = droid.video.images[t-1:t].cpu().numpy()
    disps_up = droid.video.disps_up[t-1:t].cpu().numpy()
    poses = droid.video.poses[t-1:t].cpu().numpy()
    intrinsics = droid.video.intrinsics[t-1:t].cpu().numpy()
    disps = droid.video.disps[t-1:t].cpu().numpy()

    timestamp = f"pic{t:03d}"
    Path("reconstructions/{}".format(reconstruction_path+timestamp)).mkdir(parents=True, exist_ok=True)
    np.save("reconstructions/{}/tstamps.npy".format(reconstruction_path+timestamp), tstamps)
    np.save("reconstructions/{}/images.npy".format(reconstruction_path+timestamp), images)
    np.save("reconstructions/{}/disps_up.npy".format(reconstruction_path+timestamp), disps_up)
    np.save("reconstructions/{}/disps.npy".format(reconstruction_path+timestamp), disps)
    np.save("reconstructions/{}/poses.npy".format(reconstruction_path+timestamp), poses)
    np.save("reconstructions/{}/intrinsics.npy".format(reconstruction_path+timestamp), intrinsics)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--imagedir", type=str, help="path to image directory")
    # parser.add_argument("--depthdir", type=str, help="path to image directory")

    parser.add_argument("--calib", type=str, help="path to calibration file")
    parser.add_argument("--t0", default=0, type=int, help="starting frame")
    parser.add_argument("--stride", default=1, type=int, help="frame stride")

    parser.add_argument("--weights", default="droid.pth")
    parser.add_argument("--buffer", type=int, default=512)
    parser.add_argument("--image_size", default=[240, 320])
    parser.add_argument("--disable_vis", action="store_true")

    parser.add_argument("--beta", type=float, default=0.3, help="weight for translation / rotation components of flow")
    parser.add_argument("--filter_thresh", type=float, default=2.4, help="how much motion before considering new keyframe")
    parser.add_argument("--warmup", type=int, default=8, help="number of warmup frames")
    parser.add_argument("--keyframe_thresh", type=float, default=4.0, help="threshold to create a new keyframe")
    parser.add_argument("--frontend_thresh", type=float, default=16.0, help="add edges between frames whithin this distance")
    parser.add_argument("--frontend_window", type=int, default=25, help="frontend optimization window")
    parser.add_argument("--frontend_radius", type=int, default=2, help="force edges between frames within radius")
    parser.add_argument("--frontend_nms", type=int, default=1, help="non-maximal supression of edges")

    parser.add_argument("--backend_thresh", type=float, default=22.0)
    parser.add_argument("--backend_radius", type=int, default=2)
    parser.add_argument("--backend_nms", type=int, default=3)
    parser.add_argument("--upsample", action="store_true")
    parser.add_argument("--reconstruction_path", help="path to saved reconstruction")
    args = parser.parse_args()

    args.stereo = False
    torch.multiprocessing.set_start_method('spawn')

    droid = None
    start = 1
    update = 0
    t = 0
    # need high resolution depths
    if args.reconstruction_path is not None:
        args.upsample = True

    tstamps = []
    image_list = []
    # depth_list = []
    # for (t, image, intrinsics, waiting_image, image_list) in tqdm(rgb_online(args.imagedir, args.calib, args.stride, start, update, t, image_list)):
    for (t, image, depth, intrinsics, waiting_image, image_list) in tqdm(rgbd_online(args.imagedir, args.calib, args.stride, start, update, t, image_list)):
        if t < args.t0:
            continue

        if not args.disable_vis:
            show_image(image[0])

        if droid is None:
            args.image_size = [image.shape[2], image.shape[3]]
            droid = Droid(args)
        
        # droid.track(t, image, intrinsics=intrinsics)
        droid.track(t, image, depth, intrinsics=intrinsics)
        start = 0
        
        if (t-7) == 7:
            if args.reconstruction_path is not None:
                save_initialization_reconstruction(droid, args.reconstruction_path)
        if (t-7) >= 8:
            if args.reconstruction_path is not None:
                save_reconstruction(droid, args.reconstruction_path)
        # if (t+1) % 8 == 0:
        #     if args.reconstruction_path is not None:
        #         save_initialization_reconstruction(droid, args.reconstruction_path)
        
            
        while(waiting_image):
            print("--------------waiting image!!!--------------")
            update = 1
            t, image, depth, intrinsics, waiting_image, image_list = next(rgbd_online(args.imagedir, args.calib, args.stride, start, update, t, image_list))

        

    # traj_est = droid.terminate(rgb_stream(args.imagedir, args.calib, args.stride))
    # traj_est = droid.terminate(rgbd_stream(args.imagedir, args.calib, args.stride))

