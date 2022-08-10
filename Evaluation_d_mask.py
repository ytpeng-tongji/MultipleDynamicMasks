import os
import cv2
import numpy as np
import torch
from find_high_activation import find_high_activation_mask
from Evaluation import Dice, IoU, PPV, Sensitivity
from utils.save import CsvSave, CsvCreate
from utils.helpers import makedir
import datetime

model_name = ''
mask_path = './densenet_mdm' + model_name
gt_path = './save_segmentations_full_crop'
percentile_list = [80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]
base_mask_size_list = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 32]
img_size = 224
result_path = './Result' + model_name
time_path = str(datetime.datetime.now()).replace(':', '.')
original_log = 'x0'
edge_log = 'xedge0'
mask_log = 'mask'
mix_log = 'xmix0'
class_path = os.listdir(mask_path)
makedir(result_path)

Result_Path = [result_path, time_path]
from settings import evaluation_
CsvCreate(Path_list=Result_Path, Info_list=evaluation_)

for percentile in percentile_list:

    all_mask_dice = 0
    all_mask_iou = 0
    all_mask_ppv = 0
    all_mask_sensitivity = 0
    num = 0

    for every_class_path in class_path:
        img_path = os.listdir(os.path.join(mask_path, every_class_path))
        for every_img_path in img_path:
            single_img_path = os.path.join(mask_path, every_class_path, every_img_path)
            gt_img = cv2.imread(os.path.join(gt_path, every_class_path, every_img_path.split('.')[0] + '.png'), cv2.IMREAD_GRAYSCALE)/255
            mask_img = cv2.imread(os.path.join(single_img_path, mask_log + '.jpg'), cv2.IMREAD_GRAYSCALE)

            mask_img = mask_img - np.amin(mask_img)
            mask_img = mask_img / np.amax(mask_img)
            g = torch.tensor(mask_img)
            value, indice = torch.topk(g, 2)
            mask_img_binary = find_high_activation_mask(mask_img, percentile=percentile)
            all_mask = mask_img_binary

            mask_dice = Dice.dice_coef(all_mask, gt_img)
            mask_iou = IoU.iou_score(all_mask, gt_img)
            mask_ppv = PPV.ppv(all_mask, gt_img)
            mask_sensitivity = Sensitivity.sensitivity(all_mask, gt_img)

            all_mask_dice += mask_dice
            all_mask_iou += mask_iou
            all_mask_ppv += mask_ppv
            all_mask_sensitivity += mask_sensitivity
            num += 1

    avg_mask_dice = all_mask_dice / num
    avg_mask_iou = all_mask_iou / num
    avg_mask_ppv = all_mask_ppv / num
    avg_mask_sensitivity = all_mask_sensitivity / num

    print("percentile = ", percentile)
    print("avg_mask_dice:{}, avg_mask_iou:{}, avg_mask_ppv:{}, avg_mask_sensitivity:{}".format(avg_mask_dice, avg_mask_iou, avg_mask_ppv, avg_mask_sensitivity))
    all_result = [str(percentile), str(num), avg_mask_dice, avg_mask_iou, avg_mask_ppv, avg_mask_sensitivity]
    CsvSave(Path_list=Result_Path, Info_list=all_result)