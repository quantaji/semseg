import os
import shutil

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm, trange

original_scannet_path = '/cluster/project/cvg/weders/data/scannet/'
target_scannet_path = '/scratch/scannet-pspnet-finetune'
target_scannet_path = os.path.abspath(target_scannet_path)


os.makedirs(target_scannet_path, exist_ok=True)
os.makedirs(target_scannet_path + '/train', exist_ok=True)
os.makedirs(target_scannet_path + '/list', exist_ok=True)
os.makedirs(target_scannet_path + '/train/color', exist_ok=True)
os.makedirs(target_scannet_path + '/train/label', exist_ok=True)
os.makedirs(target_scannet_path + '/val', exist_ok=True)
os.makedirs(target_scannet_path + '/val/color', exist_ok=True)
os.makedirs(target_scannet_path + '/val/label', exist_ok=True)


# train
train_scans_list = os.listdir(os.path.join(original_scannet_path, 'scans'))
with open(target_scannet_path + '/list/training.txt', 'w') as f:
    for scan_name in train_scans_list:
        scan_path = os.path.join(original_scannet_path, 'scans', scan_name)
        
        color_path = os.path.join(scan_path, 'data/color')
        color_img_list = os.listdir(color_path)
        color_img_list.sort(key=lambda e: int(e[:-4]))

        label_path = os.path.join(scan_path, 'label-proc')
        label_img_list = os.listdir(label_path)
        label_img_list.sort(key=lambda e: int(e[:-4]))

        num_scans = len(label_img_list)

        for i in trange(num_scans):

            original_color_img = os.path.join(color_path, color_img_list[i])
            original_label_img = os.path.join(label_path, label_img_list[i])

            file_name = scan_name + '_' + 'scan{:>06d}'.format(i)

            tgt_c_rel_name = os.path.join('train/color', file_name + '.' + original_color_img.split('.')[-1])
            target_color_pth = os.path.join(target_scannet_path, tgt_c_rel_name)
            tgt_l_rel_name = os.path.join('train/label', file_name + '.' + original_label_img.split('.')[-1])
            target_label_pth = os.path.join(target_scannet_path, tgt_l_rel_name)

            # copy the original color image
            shutil.copy(original_color_img, target_color_pth)
            shutil.copy(original_label_img, target_label_pth)
            # map the semantic segmentation to nyu40
            # label = cv2.imread(original_label_img, cv2.IMREAD_GRAYSCALE)
            # print(label.max())
            # label = mapping(label)
            # cv2.imwrite(target_label_pth, label)
            f.write(f'{tgt_c_rel_name} {tgt_l_rel_name}\n')

            # break
            
