from flask import Flask, request, jsonify
from gevent import pywsgi
import numpy as np
from PIL import Image
import cv2
import os

nerf_server = Flask(__name__)

def run_server():
    # must use absolute ip address not localhost
    server = pywsgi.WSGIServer(('127.0.0.1', 7400), nerf_server)
    server.serve_forever()


@nerf_server.route('/get_droid_data/', methods=['POST'])
def get_droid_data():
    global index
    if request.method == 'POST':
        received_data = request.get_json()
        pose = np.array(received_data['pose'])
        rgb_img = np.array(received_data['rgb'], dtype=np.float32)
        if len(rgb_img.shape) == 4:
            rgb_img = rgb_img.transpose(0, 2, 3, 1)[..., ::-1]
        else:
            rgb_img = rgb_img.transpose(1, 2, 0)[..., ::-1]
        os.makedirs('test_droid_data', exist_ok=True)
        if index == 0:
            for i in range(8):
                rgb_img_np = rgb_img[i]
                rgb_img_np = cv2.cvtColor(rgb_img_np, cv2.COLOR_RGB2BGR)
                np.savetxt(f'test_droid_data/{i}_pose.txt', pose[i])
                cv2.imwrite(f"test_droid_data/{i}_main.png", rgb_img[i])
            index += 7
        else:
            rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
            np.savetxt(f'test_droid_data/{index}_pose.txt', pose)
            cv2.imwrite(f"test_droid_data/{index}_main.png", rgb_img)
        index += 1
        return jsonify({'status': 'transfer success'})
    
if __name__ == '__main__':
    index = 0
    run_server()