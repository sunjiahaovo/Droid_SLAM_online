import os
import glob
import shutil

data_dir = '/home/perple/Public/DROID-SLAM/data/jiahao_room1'
rgb_data = glob.glob(os.path.join(data_dir, 'frame*.png'))
rgb_data.sort()
print(len(rgb_data))
os.makedirs('/home/perple/Public/DROID-SLAM/data/jiahao_room1_20',exist_ok=True)

for i, rgb in enumerate(rgb_data):
    if (i + 5) % 20 == 0:
        shutil.copy(rgb, os.path.join('/home/perple/Public/DROID-SLAM/data/jiahao_room1_20', os.path.basename(rgb)))