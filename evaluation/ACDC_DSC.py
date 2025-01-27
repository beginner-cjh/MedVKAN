# -*- coding: utf-8 -*-  
"""  
Created on [modified date]  

@author: [your name]  
"""

import numpy as np
import nibabel as nb
import os
from collections import OrderedDict
import pandas as pd
from tqdm import tqdm

# 假设 SurfaceDice.py 提供了必要的函数，这里我们不需要修改它  
from SurfaceDice import compute_surface_distances, compute_surface_dice_at_tolerance, compute_dice_coefficient

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--gt_path', type=str, default='')
parser.add_argument('--seg_path', type=str, default='')
#parser.add_argument('--save_path', type=str, default='')

args = parser.parse_args()

gt_path = args.gt_path
seg_path = args.seg_path
#save_path = args.save_path

filenames = os.listdir(seg_path)
filenames = [x for x in filenames if x.endswith('.nii.gz')]
filenames = [x for x in filenames if os.path.exists(os.path.join(seg_path, x))]
filenames.sort()

seg_metrics = OrderedDict()
seg_metrics['Name'] = list()
label_tolerance = OrderedDict({'Liver': 5, 'Spleen': 3, 'Pancreas': 5})  # 仅保留3个器官  
for organ in label_tolerance.keys():
    seg_metrics['{}_DSC'.format(organ)] = list()


def find_lower_upper_zbound(organ_mask):
    """  
    找到器官掩码在z轴上的上下界。  
    """
    organ_mask = np.uint8(organ_mask)
    assert np.max(organ_mask) == 1, "mask label error!"
    z_index = np.where(organ_mask > 0)[2]
    z_lower = np.min(z_index)
    z_upper = np.max(z_index)
    return z_lower, z_upper


for name in tqdm(filenames):
    seg_metrics['Name'].append(name)
    # 加载ground truth和segmentation  
    gt_nii = nb.load(os.path.join(gt_path, name))
    case_spacing = gt_nii.header.get_zooms()
    gt_data = np.uint8(gt_nii.get_fdata())
    seg_data = np.uint8(nb.load(os.path.join(seg_path, name)).get_fdata())

    for i, organ in enumerate(label_tolerance.keys(), 1):
        if np.sum(gt_data == i) == 0 and np.sum(seg_data == i) == 0:
            DSC_i = 1
            NSD_i = 1
        elif np.sum(gt_data == i) == 0 and np.sum(seg_data == i) > 0:
            DSC_i = 0
            NSD_i = 0
        else:
            # 对于这3个器官，我们不需要特别的z轴切片评估  
            organ_i_gt, organ_i_seg = gt_data == i, seg_data == i
            DSC_i = compute_dice_coefficient(organ_i_gt, organ_i_seg)
        seg_metrics['{}_DSC'.format(organ)].append(round(DSC_i, 4))

    dataframe = pd.DataFrame(seg_metrics)
    # 更新保存路径以包含3分割的信息  
    # 例如，可以添加前缀或后缀来区分文件  
    #save_path_updated = save_path.replace('.csv', '_3seg.csv')  # 假设save_path以.csv结尾
    #dataframe.to_csv(save_path_updated, index=False)

# 注意：下面的代码块现在应该在循环外部或根据需要进行调整，  
# 因为我们已经在每次迭代中保存了数据帧。  
# 如果只想保存最终的平均值，可以注释掉数据帧的每次迭代保存，  
# 并在循环后计算平均值。  

case_avg_DSC = dataframe.mean(axis=0, numeric_only=True)
print(20 * '>')
print(f'Average DSC for {os.path.basename(seg_path)}: {case_avg_DSC.mean()}')
print(20 * '<')

# 由于我们在每次迭代中都保存了数据帧，上面的平均值计算应该被移除或放在其他地方。  
# 例如，如果你想在所有文件处理完毕后计算一次平均值，你可以这样做：  
# 1. 注释掉循环内部的 dataframe.to_csv 调用。  
# 2. 在循环外部，使用所有文件的 seg_metrics 计算平均值。  
# 3. 然后保存最终的数据帧或打印平均值。
