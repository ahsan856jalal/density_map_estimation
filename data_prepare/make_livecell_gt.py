import h5py
import scipy.io as io
import PIL.Image as Image
import numpy as np
import os
import glob
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from matplotlib import cm as CM
from utils import gaussian_filter_density
from tqdm import tqdm
import cv2

# set the root to the dataset you download
# ensure the shorter side > min_size
min_size = 384

root = "/content/drive/MyDrive/MARUNet/live_cell/skov3_data/"

save_root_dir='/content/drive/MyDrive/MARUNet/datasets/skov3_data'
train_save = save_root_dir +'/train/'
test_save = save_root_dir+'/test/'
if not os.path.exists(train_save):
    os.makedirs(train_save)
if not os.path.exists(test_save):
    os.makedirs(test_save)
# now generate the ground truth density maps
train_dir = root+'train/'
test_dir = root+'test/'
path_sets = [train_dir, test_dir]

img_paths = []
for path in path_sets:
    for img_path in glob.glob(os.path.join(path, '*.tif')):
        img_paths.append(img_path)

for img_path in tqdm(img_paths):
    mat = io.loadmat(img_path.replace('.tif','.mat'))
    img = cv2.imread(img_path)
    img_save_path = os.path.join(train_save, img_path.split('/')[-1]) if 'train' in img_path else os.path.join(test_save, img_path.split('/')[-1])
    img_h, img_w = img.shape[0:2]
    # ensure the shorter side > min_size
    if img_h<img_w:
        if img_h<min_size:
            # up sample ratio is int
            ratio = min_size/img_h
            new_w, new_h = int(ratio*img_w), min_size
        else:
            ratio = 1
            new_w, new_h = img_w, img_h
    else:
        if img_w<min_size:
            ratio = min_size/img_w
            new_w, new_h = min_size, int(ratio*img_h)
        else:
            ratio = 1
            new_w, new_h = img_w, img_h
    if ratio == 1:
        cv2.imwrite(img_save_path, img)
    
    else:
        img_resize = cv2.resize(img, (new_w, new_h))
        cv2.imwrite(img_save_path, img_resize)
        print('img:',img_path.split('/')[-1],', img_h, img_w:',img_h, ', ', img_w, ', new_h, new_w:', new_h, ', ', new_w)
    k = np.zeros((new_h, new_w))
    image_info = mat['image_info']
    gt = image_info[0, 0]['location']
    # gt = mat["image_info"][0, 0][0, 0][0]
    for position in gt:
        x = int(position[0]*ratio)
        y = int(position[1]*ratio)
        if x >= new_w or y >= new_h:
            continue
        k[y, x]=1
    # if dataset=='sha':
    k = gaussian_filter_density(k,4,10)
        #k = gaussian_filter(k, 4)
    # if dataset=='shb':
    #     k = gaussian_filter(k, 15, truncate=4.0)
    
    with h5py.File(img_save_path.replace('.tif', '.h5'), 'w') as hf:
        hf['density'] = k