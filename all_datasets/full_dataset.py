import os
import shutil
import tqdm
import time

all_images = []
all_points = []

root = '/home/dilip/Code/ml/all_datasets/'
a = []
for dirs, subdirs, files in os.walk(root):
    for f in files:
        if f.endswith('.jpg') or f.endswith('.png'):
            img = os.path.join(dirs, f)
            all_images.append(img)
        if f.endswith('.pts'):
            pts = os.path.join(dirs,f)
            all_points.append(pts)
            a.append(f)

data = '/home/dilip/Code/ml/data/'

for src_x in tqdm.tqdm(all_images):
    nx = src_x[33:].replace('/','-')
    if nx.startswith('300vw'):
        nx = nx.replace('image-','')
    dst_x = os.path.join(data, nx)

    # shutil.copy(src_x, dst_x)

    time.sleep(1)
    print(f'{src_x}, {dst_x}')

for src_y in tqdm.tqdm(all_points):
    ny = src_y[33:].replace('/','-')
    if ny.startswith('300vw'):
        ny = ny.replace('annot-','')
    dst_y = os.path.join(data, ny)

    # shutil.copy(src_y, dst_y)

    time.sleep(1)
    print(f'{src_y}, {dst_y}')
