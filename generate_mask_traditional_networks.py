import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from find_high_activation import find_high_activation_mask
from Evaluation import Dice, IoU, PPV, Sensitivity
from utils.save import CsvSave, CsvCreate
from utils.helpers import makedir
import datetime

mdm_save = './densenet_mdm'
original_path = './bird_datasets'
mask_path = './densenet_full_datasets'
gt_path = './save_full_segmentations_crop'
percentile_list = [1]
base_mask_size_list = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 29, 30, 32, 56, 112]
img_size = 224
threshold = 5
result_path = './Result'
time_path = str(datetime.datetime.now()).replace(':', '.')
original_log = 'x'
edge_log = 'xedge'
mask_log = 'xmask'
mix_log = 'xmix'
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
            all_mask = np.zeros([img_size, img_size])
            real_all_mask = np.zeros([224, 224])

            for base_size in base_mask_size_list:
                mask_img = cv2.imread(os.path.join(single_img_path, str(base_size) + mask_log + '.png'), cv2.IMREAD_GRAYSCALE)
                mix_img = cv2.imread(os.path.join(single_img_path, str(base_size) + mix_log + '.png'))[..., ::-1]

                mask_img = mask_img - np.amin(mask_img)
                mask_img = mask_img / np.amax(mask_img)
                mask_img_binary = find_high_activation_mask(mask_img, percentile=percentile)

                real_all_mask += mask_img
                all_mask += mask_img_binary

            heat_map = all_mask
            heat_map = heat_map - np.amin(heat_map)
            heat_map = heat_map / np.amax(heat_map)
            heat_map = np.reshape(heat_map, (224, 224, 1))
            heat_map = heat_map.repeat(3, 2)

            original_img = cv2.imread(os.path.join(original_path, every_class_path, every_img_path))[..., ::-1]
            original_heat_map_ = heat_map * original_img
            original_heat_map_ = original_heat_map_.astype('uint8')

            all_mask[all_mask < threshold] = 0
            all_mask[all_mask >= threshold] = 1
            real_all_mask = real_all_mask * all_mask - threshold
            real_heat_map = real_all_mask
            real_heat_map = real_heat_map - np.amin(real_heat_map)
            real_heat_map = real_heat_map / np.amax(real_heat_map)

            heatmaps = cv2.applyColorMap(np.uint8(255 * real_heat_map), cv2.COLORMAP_JET)
            heatmaps = np.float32(heatmaps) / 255
            heatmaps = heatmaps[..., ::-1]
            overlayed_original_img_j = 0.5 * original_img/255 + 0.3 * heatmaps
            overlayed_original_img_j = (overlayed_original_img_j - np.min(overlayed_original_img_j)) / (np.max(overlayed_original_img_j) - np.min(overlayed_original_img_j))

            real_heat_map = np.reshape(real_heat_map, (224, 224, 1))
            real_heat_map = real_heat_map.repeat(3, 2)

            real_original_heat_map_ = real_heat_map * original_img
            real_original_heat_map_ = real_original_heat_map_.astype('uint8')

            mask_dice = Dice.dice_coef(all_mask, gt_img)
            mask_iou = IoU.iou_score(all_mask, gt_img)
            mask_ppv = PPV.ppv(all_mask, gt_img)
            mask_sensitivity = Sensitivity.sensitivity(all_mask, gt_img)

            all_mask_dice += mask_dice
            all_mask_iou += mask_iou
            all_mask_ppv += mask_ppv
            all_mask_sensitivity += mask_sensitivity
            num += 1
            print(num)

            makedir(os.path.join(mdm_save, every_class_path, every_img_path))
            plt.imsave(os.path.join(mdm_save, every_class_path, every_img_path, 'camcam.jpg'), overlayed_original_img_j)
            plt.imsave(os.path.join(mdm_save, every_class_path, every_img_path, 'mask.jpg'), real_heat_map[:, :, 0])

    avg_mask_dice = all_mask_dice / num
    avg_mask_iou = all_mask_iou / num
    avg_mask_ppv = all_mask_ppv / num
    avg_mask_sensitivity = all_mask_sensitivity / num

    print("percentile = ", percentile)
    print("avg_mask_dice:{}, avg_mask_iou:{}, avg_mask_ppv:{}, avg_mask_sensitivity:{}".format(avg_mask_dice, avg_mask_iou, avg_mask_ppv, avg_mask_sensitivity))
    all_result = [str(percentile), str(num), avg_mask_dice, avg_mask_iou, avg_mask_ppv, avg_mask_sensitivity]
    CsvSave(Path_list=Result_Path, Info_list=all_result)
