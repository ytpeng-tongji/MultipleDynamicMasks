import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch
from analysis_settings import img_size_, mean, std, dynamic_img_batch_, dynamic_mask_batch_, bb_img_path, base_mask_height_width_list_, \
    dynamic_mask_lrs, iteration_epoch_, iteration_epoch_min_, mask_optimizer_lr_, base_mask_size_list_, img_mask_root_path1_, original_img_path1_, result_path1_, pathology_path1_
import dynamic_maskpair
from utils.helpers import read_img_label
import os
from utils.helpers import makedir
import matplotlib.pyplot as plt
import torch.nn as nn
import cv2
import datetime
from utils.save import CsvSave, TxtCreate, CsvCreate
import csv
import model # pre-trained model architecture

img_size = img_size_
original_img_path = original_img_path1_
result_path = result_path1_
img_mask_root_paths = img_mask_root_path1_
pathology_label_img_path = pathology_path1_
normalize = transforms.Normalize(mean=mean, std=std)
dynamic_img_batch = dynamic_img_batch_
dynamic_mask_batch = dynamic_mask_batch_
num_prototypes = 42
num_classes = 14
episilon = 1e-12
prototype_class_identity = torch.zeros(num_prototypes, num_classes)
num_prototypes_per_class = int(num_prototypes // num_classes)
label_dictionary = {'Atelectasis': 0, 'Cardiomegaly': 1, 'Effusion': 2, 'Infiltrate': 3, 'Mass': 4, 'Nodule': 5, 'Pneumonia': 6, 'Pneumothorax': 7}
for j in range(num_prototypes):
    prototype_class_identity[j, j // num_prototypes_per_class] = 1

img_name_list = os.listdir(bb_img_path)
real_label = []
for i in img_name_list:
    real_label.append(read_img_label(os.path.join(bb_img_path, i)))

def GetLabel():
    bb_label = './NIH_label/BBox_List_2017.csv'
    label_class = []
    all_class = []
    NameToClass = {}
    Name_list = []
    with open(bb_label, 'r') as f:
        csv_reader = csv.reader(f)
        num = 0

        for line in csv_reader:
            if (num != 0):
                label_class.append(line[1])
                NTC = {line[0]: line[1]}
                NameToClass.update(NTC)
                Name_list.append(line[0])
                if (line[1] in all_class):
                    pass
                else:
                    all_class.append(line[1])
            num += 1

    return label_class, NameToClass, Name_list

def Dynamic_Mask_bb(ppnet):
    original_img_dataset = datasets.ImageFolder(
        original_img_path,
        transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor(),
            normalize,
        ]))

    original_img_loader = torch.utils.data.DataLoader(
        original_img_dataset, batch_size=dynamic_img_batch, shuffle=False,
        num_workers=4, pin_memory=False)

    Label_list, NameToClass, Name_list = GetLabel()
    LocalNameList = os.listdir(os.path.join(original_img_path, 'bbox_valid'))

    label_img_list = os.listdir(pathology_label_img_path)
    img_mask_root_path = img_mask_root_paths
    current_time = str(datetime.datetime.now()).replace(':', '.')
    makedir(img_mask_root_path)

    Txt_Path_list = [img_mask_root_path, current_time]
    Info_list = []
    with open("analysis_settings.py", "r", encoding="UTF-8") as f:
        setting_info = f.read()
    Info_list += [setting_info]
    TxtCreate(Path_list=Txt_Path_list, Info_list=Info_list)
    img_mask_root_path = os.path.join(img_mask_root_path, current_time)

    Result_Path = [result_path, current_time]
    makedir(result_path)
    from analysis_settings import IsPositivi_
    CsvCreate(Path_list=Result_Path, Info_list=IsPositivi_)

    for batch_idx, (x, _) in enumerate(original_img_loader):
        x = x[:, 0:1, :, :].cuda()
        total_batch = batch_idx * x.shape[0]
        total_img_start = batch_idx * x.shape[0]
        total_img_end = (batch_idx + 1) * x.shape[0]
        real_label_name = Label_list[total_img_start:total_img_end]
        real_label = []
        for single_label_name in real_label_name:
            real_label.append(label_dictionary[single_label_name])
        output, min_distances = ppnet_multi(x)
        predicted_positive = []
        threshold = 0.1

        for k in range(output.shape[0]):
            if (output[k][real_label[k]] > threshold):
                predicted_positive.append(True)
                all_result = [str(batch_idx * x.shape[0] + k), LocalNameList[batch_idx * x.shape[0]],
                              Label_list[batch_idx * x.shape[0]], 'True', output[k][real_label[k]].item()]
                CsvSave(Path_list=Result_Path, Info_list=all_result)
            else:
                predicted_positive.append(False)
                all_result = [str(batch_idx * x.shape[0] + k), LocalNameList[batch_idx * x.shape[0]],
                              Label_list[batch_idx * x.shape[0]], 'False', output[k][real_label[k]].item()]
                CsvSave(Path_list=Result_Path, Info_list=all_result)

        _, predicted = torch.max(output.data, 1)
        _, torch_x_dist = ppnet.push_forward(x)
        prototype_batch = np.zeros([x.shape[0], num_prototypes_per_class])

        for i in range(x.shape[0]):
            first_indice = int(torch.argmax(prototype_class_identity.t()[real_label[i]]).item())
            prototype_batch[i] = [first_indice, first_indice + 1, first_indice + 2]

        for i in range(x.shape[0]):
            img_num = total_batch + i
            x_img_i = cv2.imread(os.path.join(pathology_label_img_path, label_img_list[img_num]), cv2.IMREAD_GRAYSCALE)
            base_mask_size_list = base_mask_size_list_
            base_mask_height_width_list = base_mask_height_width_list_

            for base_mask_num in range(len(base_mask_size_list)):
                for j in range(num_prototypes_per_class):
                    mask_img_path = os.path.join(img_mask_root_path, LocalNameList[batch_idx * x.shape[0] + i])
                    makedir(mask_img_path)
                    d_mask = dynamic_maskpair.Dynamic_MaskPair(img_size=img_size, base_mask_height_width=base_mask_height_width_list[base_mask_num])
                    d_mask = d_mask.cuda()
                    original_min = torch_x_dist[int(i), int(prototype_batch[i][j]), :, :]
                    min_activation = torch.min(original_min)
                    min_activation_index = [int(torch.argmin(original_min) // original_min.shape[1]),
                                            int(torch.argmin(original_min) % original_min.shape[1])]
                    min_activation = torch.full([d_mask.mask_batch], min_activation.item()).cuda()

                    iteration_epoch = iteration_epoch_
                    iteration_epoch_min = iteration_epoch_min_
                    mask_optimizer_lr = mask_optimizer_lr_

                    base_mask_height = d_mask.base_mask_height
                    base_mask_width = d_mask.base_mask_width
                    d_mask = torch.nn.DataParallel(d_mask)
                    mask_optimizer_specs = [{'params': d_mask.module.mask, 'lr': mask_optimizer_lr}]
                    optimizer = torch.optim.Adam(mask_optimizer_specs)

                    for epoch in range(iteration_epoch):
                        if (epoch > iteration_epoch_min):
                            break
                        else:
                            x_mask = d_mask(x[i])
                            _, torch_x_mask_dist = ppnet.push_forward(x_mask)
                            mask_min = torch_x_mask_dist[:, int(prototype_batch[i][j]), :, :]
                            mask_min_activation = mask_min[:, min_activation_index[0], min_activation_index[1]].cuda()
                            mse_loss = (mask_min_activation - min_activation) ** 2
                            mse_loss = mse_loss.cuda()
                            mean_l1_loss = d_mask.module.mask.norm(p=1, dim=(1, 2)) / (base_mask_height * base_mask_width)
                            loss = dynamic_mask_lrs['mse'] * mse_loss + dynamic_mask_lrs['l1'] * mean_l1_loss
                            loss = loss.sum()
                            loss = loss.cuda()

                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()

                    upsample = nn.Upsample(size=(img_size, img_size), mode='bilinear', align_corners=True)
                    real_mask = upsample(d_mask.module.mask.unsqueeze(0)).squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
                    real_mask_max = np.max(real_mask)
                    real_mask_min = np.min(real_mask)
                    mask_01 = (real_mask - real_mask_min) / (real_mask_max - real_mask_min)
                    heatmap = cv2.applyColorMap(np.uint8(255 * mask_01), cv2.COLORMAP_JET)
                    heatmap = np.float32(heatmap) / 255
                    heatmap = heatmap[..., ::-1]
                    x_img_i_rgb = np.repeat(np.reshape(x_img_i, (512, 512, 1)), repeats=3, axis=2)
                    overlayed_original_img_j = 0.5 * x_img_i_rgb / 255 + 0.3 * heatmap
                    overlayed_original_img_j = (overlayed_original_img_j - np.min(overlayed_original_img_j)) / (np.max(overlayed_original_img_j) - np.min(overlayed_original_img_j))
                    x_img_i = np.reshape(x_img_i, (512, 512, 1))
                    mix_img = mask_01 * x_img_i
                    mix_img = (mix_img - np.min(mix_img)) / (np.max(mix_img) - np.min(mix_img))
                    mix_img = np.repeat(mix_img, repeats=3, axis=2)
                    plt.imsave(os.path.join(mask_img_path, str(base_mask_height_width_list[base_mask_num]) + 'x' + 'binary_mask_original' + str(j) + '.png'), mix_img)
                    plt.imsave(os.path.join(mask_img_path, str(base_mask_height_width_list[base_mask_num]) + 'x' + 'binary_mask' + str(j) + '.png'), real_mask[:, :, 0])
                    plt.imsave(os.path.join(mask_img_path, str(base_mask_height_width_list[base_mask_num]) + 'x' + 'heatmap_original' + str(j) + '.png'), overlayed_original_img_j)

if __name__ == '__main__':
    # path of pre-trained model
    load_model_path = './pretrained_models/'
    ppnet = torch.load(load_model_path)
    ppnet = ppnet.cuda()
    ppnet_multi = torch.nn.DataParallel(ppnet)
    Dynamic_Mask_bb(ppnet)

