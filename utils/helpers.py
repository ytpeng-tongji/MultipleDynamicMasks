import os
import torch
import numpy as np
import pandas as pd

def list_of_distances(X, Y):
    return torch.sum((torch.unsqueeze(X, dim=2) - torch.unsqueeze(Y.t(), dim=0)) ** 2, dim=1)

def read_img_label(path):
    label_dictionary = {'Atelectasis': 0, 'Cardiomegaly': 1, 'Effusion': 2, 'Infiltrate': 3, 'Mass': 4, 'Nodule': 5,
                        'Pneumonia': 6, 'Pneumothorax': 7, 'Consolidation': 8, 'Edema': 9, 'Emphysema': 10,
                        'Fibrosis': 11, 'Pleural_Thickening': 12, 'Hernia': 13}

    image_name = str(os.path.split(path)[1])
    bb_list_path = './Active_test/nih_bbox/BBox_List_2017.csv'
    label = pd.read_csv(bb_list_path).values
    name_list = label[:, 0]
    indice_set = {}
    for i in range(len(name_list)):
        indice_set.update({name_list[i]: i})

    pathology_name = label[indice_set[image_name]][1]
    pathology_label = label_dictionary[pathology_name]

    return pathology_label

def make_one_hot(target, target_one_hot):
    target = target.view(-1, 1)
    target_one_hot.zero_()
    target_one_hot.scatter_(dim=1, index=target, value=1.)

def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def print_and_write(str, file):
    print(str)
    file.write(str + '\n')

def find_high_activation_crop(activation_map, percentile=95):
    threshold = np.percentile(activation_map, percentile)
    mask = np.ones(activation_map.shape)
    mask[activation_map < threshold] = 0
    lower_y, upper_y, lower_x, upper_x = 0, 0, 0, 0
    for i in range(mask.shape[0]):
        if np.amax(mask[i]) > 0.5:
            lower_y = i
            break

    for i in reversed(range(mask.shape[0])):
        if np.amax(mask[i]) > 0.5:
            upper_y = i
            break

    for j in range(mask.shape[1]):
        if np.amax(mask[:, j]) > 0.5:
            lower_x = j
            break

    for j in reversed(range(mask.shape[1])):
        if np.amax(mask[:, j]) > 0.5:
            upper_x = j
            break

    return lower_y, upper_y + 1, lower_x, upper_x + 1

