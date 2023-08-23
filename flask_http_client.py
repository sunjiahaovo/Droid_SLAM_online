import requests
import time
import glob

# def send_image_to_server(image_path, stop_flag, server_url):
def send_image_to_server(image_path, depth_path, stop_flag, server_url):
    with open(image_path, 'rb') as rgb_file:
        rgb_data = rgb_file.read()
    # with depth
    with open(depth_path, 'rb') as depth_file:
        depth_data = depth_file.read()
    # with depth
    files = {'rgb': (image_path, rgb_data, 'image/png'),
             'depth': (depth_path, depth_data, 'image/png')}
    # without depth
    # files = {'rgb': (image_path, rgb_data, 'image/png')}    
    # response = requests.post(server_url, data={"flying_finished":stop_flag}, files=files)
    print(response.text, end='')  # 输出服务器返回的响应内容

if __name__ == '__main__':
    server_url = 'http://127.0.0.1:7300/get_flying_data/'
    # image_path = glob.glob('/home/perple/Public/DROID-SLAM/data/bear_room_ipad_nopitch_photo/images/frame*.png')
    image_path = glob.glob('/home/perple/Public/DROID-SLAM/data/bear_room_realsense_online_ellipse_noWB/frame*.png')
    image_path.sort()
    # with depth
    depth_path = glob.glob('/home/perple/Public/DROID-SLAM/data/bear_room_realsense_online_ellipse_noWB/depth*.png')
    assert len(image_path) == len(depth_path)
    depth_path.sort()
    for i,(rgb, depth) in enumerate(zip(image_path, depth_path)):
        stop_flag = False if i < len(image_path) - 1 else True
        send_image_to_server(rgb, depth, stop_flag, server_url)
        time.sleep(3)
    # for i, rgb in enumerate(image_path):
    #     stop_flag = False if i < len(image_path) - 1 else True
    #     send_image_to_server(rgb, stop_flag, server_url)
    #     time.sleep(3)
