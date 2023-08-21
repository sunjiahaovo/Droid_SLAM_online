import sys
sys.path.append('droid_slam')

from tqdm import tqdm
import numpy as np
import torch

import cv2
from PIL import Image
import skimage
import os

import time
import argparse


from torch.multiprocessing import Process
from droid import Droid

import torch.nn.functional as F

from flask import Flask, request, jsonify
from gevent import pywsgi
import requests
import json
droid_site = Flask(__name__)

from threading import Thread

def show_image(image):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey(1)


def save_reconstruction(droid, reconstruction_path):
    from pathlib import Path
    import random
    import string

    t = droid.video.counter.value
    tstamps = droid.video.tstamp[:t].cpu().numpy()
    images = droid.video.images[:t].cpu().numpy()
    disps_up = droid.video.disps_up[:t].cpu().numpy()
    poses = droid.video.poses[:t].cpu().numpy()
    intrinsics = droid.video.intrinsics[:t].cpu().numpy()
    disps = droid.video.disps[:t].cpu().numpy()

    Path("reconstructions/{}".format(reconstruction_path)).mkdir(parents=True, exist_ok=True)
    np.save("reconstructions/{}/tstamps.npy".format(reconstruction_path), tstamps)
    np.save("reconstructions/{}/images.npy".format(reconstruction_path), images)
    np.save("reconstructions/{}/disps_up.npy".format(reconstruction_path), disps_up)
    np.save("reconstructions/{}/disps.npy".format(reconstruction_path), disps)
    np.save("reconstructions/{}/poses.npy".format(reconstruction_path), poses)
    np.save("reconstructions/{}/intrinsics.npy".format(reconstruction_path), intrinsics)


def parse_args():
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
    return args



def droid_processing(t, rgb_img, depth_img, intrinsic):
    show_image(rgb_img[0])
    droid.track(t, rgb_img, depth_img, intrinsics=intrinsic)


@droid_site.route('/get_flying_data/', methods=['POST'])
def get_flying_data():
    global index
    if request.method == 'POST':
        rgb_img = request.files.get("rgb", None)
        # depth_img = request.files.get("depth", None)
        rgb_img = np.array(Image.open(rgb_img))
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
        # depth_img = np.array(skimage.io.imread(depth_img))

        os.makedirs("test_flying_data", exist_ok=True)
        cv2.imwrite(f"test_flying_data/{index}_main.png", rgb_img)
        # cv2.imwrite(f"test_flying_data/{index}_depth.png", depth_img)
        
        h0, w0, _ = rgb_img.shape
        h1 = int(h0 * np.sqrt((384 * 512) / (h0 * w0)))
        w1 = int(w0 * np.sqrt((384 * 512) / (h0 * w0)))   
       
        rgb_img = cv2.resize(rgb_img, (w1, h1))
        rgb_img = rgb_img[:h1-h1%8, :w1-w1%8]
        rgb_img = torch.as_tensor(rgb_img).permute(2, 0, 1)

        # depth_img = depth_img.astype(np.float32)
        # depth_img = torch.as_tensor(depth_img)
        # # for scale 
        # depth_img /= 1000
        # depth_img = F.interpolate(depth_img[None,None], (h1, w1)).squeeze()
        # depth_img = depth_img[:h1-h1%8, :w1-w1%8]
        intrinsic = torch.zeros(4, dtype=torch.float32)
        intrinsic[0::2] = droid_site.config['intrinsic'][0::2] * (w1 / w0)
        intrinsic[1::2] = droid_site.config['intrinsic'][1::2] * (h1 / h0)
        # droid_processing(index, rgb_img[None], depth_img, intrinsic)
        droid_processing(index, rgb_img[None], None, intrinsic)
        # print(droid.video.images[index].shape)
        # if index == 8:
        #     nerf_data = {
        #         "rgb" : droid.video.images[:index].cpu().numpy().tolist(),
        #         "pose": droid.video.poses[:index].cpu().numpy().tolist(),
        #     }
        #     response = requests.post('http://127.0.0.1:7400/get_droid_data/', json=nerf_data)
        #     print(response.text, end='')  # 输出服务器返回的响应内容
        # if index >= 9:
        #     nerf_data = {
        #         "rgb" : droid.video.images[index].cpu().numpy().tolist(),
        #         "pose": droid.video.poses[index].cpu().numpy().tolist(),
        #     }
            # response = requests.post('http://127.0.0.1:7400/get_droid_data/', json=nerf_data)
            # print(response.text, end='')  # 输出服务器返回的响应内容
        if index >= 8:
            nerf_data = {
                "rgb" : droid.video.images[:index].cpu().numpy().tolist(),
                "pose": droid.video.poses[:index].cpu().numpy().tolist(),
            }
            response = requests.post('http://127.0.0.1:7400/get_droid_data/', json=nerf_data)
            print(response.text, end='')  # 输出服务器返回的响应内容
        pose = droid.video.poses[:index].cpu().numpy()
        rgb = droid.video.images[:index].cpu().numpy()
        print(rgb.shape)
        os.makedirs('test_droid_data', exist_ok=True)
        np.save(f'test_droid_data/{index}_pose.npy', pose)
        np.save((f'test_droid_data/{index}_rgb.npy', rgb))
        # rgb = rgb.transpose(1, 2, 0)
        # cv2.imwrite(f"test_droid_data/{index-7}_main.png", rgb.astype(np.uint8))
        index += 1
        if request.form.get("flying_finished", None) == "True":
            save_reconstruction(droid, args.reconstruction_path)
        return jsonify({'status': 'transfer success'})
        
def run_server():
    # must use absolute ip address not localhost
    server = pywsgi.WSGIServer(('127.0.0.1', 7300), droid_site)
    # nesc
    # server = pywsgi.WSGIServer(('192.168.31.250', 7300), droid_site)
    # tplink-b6d5
    # server = pywsgi.WSGIServer(('192.168.1.112', 7300), droid_site)
    # optitrack
    # server = pywsgi.WSGIServer(('192.168.50.56', 7300), droid_site)   
    server.serve_forever()
    
if __name__ == '__main__':
    args = parse_args()
    torch.multiprocessing.set_start_method('spawn')
    droid = None
    # need high resolution depths
    if args.reconstruction_path is not None:
        args.upsample = True
    args.image_size = [384, 512]
    droid = Droid(args)
    droid_site.config['args'] = args
    droid_site.config['intrinsic'] = torch.tensor(np.loadtxt(args.calib, delimiter=" "))
    index  = 0
    run_server()
    # server_thread = Thread(target=run_server, daemon=True)
    # server_thread.start()
   
    # server_thread.join()
        
