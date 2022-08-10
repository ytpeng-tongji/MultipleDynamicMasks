import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch
from analysis_settings import img_size_, original_img_path_, mean, std, dynamic_img_batch_, dynamic_mask_batch_, dynamic_mask_lrs, base_mask_height_width_list_,\
best_loss_, best_epoch_, iteration_epoch_, iteration_epoch_min_, patient_, mask_optimizer_lr_, check_epoch_
import dynamic_maskpair
import cv2
import os
from utils.helpers import makedir
import matplotlib.pyplot as plt
import torch.nn as nn
import model # pre-trained model architecture

img_size = img_size_
original_img_path = original_img_path_
normalize = transforms.Normalize(mean=mean, std=std)
dynamic_img_batch = dynamic_img_batch_
dynamic_mask_batch = dynamic_mask_batch_
img_mask_root_path = './img_mask_root_path'
bird_label_img_path = './bird_datasets/001.Black_footed_Albatross'
num_prototypes = 2000
num_classes = 200
episilon = 1e-12
prototype_class_identity = torch.zeros(num_prototypes, num_classes)
num_prototypes_per_class = int(num_prototypes // num_classes)
for j in range(num_prototypes):
    prototype_class_identity[j, j // num_prototypes_per_class] = 1

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

    img_class = os.listdir(original_img_path)
    for batch_idx, (x, y) in enumerate(original_img_loader):
        x = x.cuda()
        output, min_distances = ppnet_multi(x)
        _, predicted = torch.max(output.data, 1)
        _, torch_x_dist = ppnet.push_forward(x)
        prototype_batch = np.zeros([x.shape[0], num_prototypes_per_class])

        for i in range(x.shape[0]):
            first_indice = int(torch.argmax(prototype_class_identity.t()[predicted[i]]).item())
            for j in range(num_prototypes_per_class):
                prototype_batch[i][j] = first_indice + j

        for i in range(x.shape[0]):
            base_mask_height_width_list = base_mask_height_width_list_
            for base_mask_num in range(len(base_mask_height_width_list)):
                for j in range(num_prototypes_per_class):
                    mask_img_path = os.path.join(img_mask_root_path, img_class[y[i].item()])
                    datasets_img_path = os.path.join(original_img_path, img_class[y[i].item()])
                    img_name = os.listdir(datasets_img_path)
                    mask_img_path = os.path.join(mask_img_path, img_name[i])
                    makedir(mask_img_path)
                    d_mask = dynamic_maskpair.Dynamic_MaskPair(img_size=img_size, base_mask_height_width=base_mask_height_width_list[base_mask_num])
                    d_mask = d_mask.cuda()
                    original_min = torch_x_dist[int(i), int(prototype_batch[i][j]), :, :]

                    min_activation = torch.min(original_min)
                    min_activation_index = [int(torch.argmin(original_min) // original_min.shape[1]), int(torch.argmin(original_min) % original_min.shape[1])]
                    min_activation = torch.full([d_mask.mask_batch], min_activation.item()).cuda()

                    best_loss = best_loss_
                    best_epoch = best_epoch_
                    iteration_epoch = iteration_epoch_
                    iteration_epoch_min = iteration_epoch_min_
                    patient = patient_
                    mask_optimizer_lr = mask_optimizer_lr_
                    check_epoch = check_epoch_

                    base_mask_height = d_mask.base_mask_height
                    base_mask_width = d_mask.base_mask_width
                    d_mask = torch.nn.DataParallel(d_mask)
                    mask_optimizer_specs = [{'params': d_mask.module.mask, 'lr': mask_optimizer_lr}]
                    optimizer = torch.optim.Adam(mask_optimizer_specs)

                    for epoch in range(iteration_epoch):
                        if((epoch - best_epoch) > patient and epoch > iteration_epoch_min):
                            break
                        else:
                            x_mask = d_mask(x[i])
                            _, torch_x_mask_dist = ppnet.push_forward(x_mask)
                            mask_min = torch_x_mask_dist[:, int(prototype_batch[i][j]), :, :]
                            mask_min_activation = mask_min[:, min_activation_index[0], min_activation_index[1]].cuda()
                            mse_loss = (mask_min_activation - min_activation)**2
                            mse_loss = mse_loss.cuda()
                            mean_l1_loss = d_mask.module.mask.norm(p=1, dim=(1, 2)) / (base_mask_height * base_mask_width)
                            loss = dynamic_mask_lrs['mse']*mse_loss + dynamic_mask_lrs['l1']*mean_l1_loss
                            loss = loss.sum()
                            loss = loss.cuda()

                            if (epoch % check_epoch == 0):
                                print("epoch = ", epoch)
                                print("mse_loss = ", mse_loss)
                                print("l1_loss = ", mean_l1_loss)
                                print("loss = ", loss)

                            if(np.min(loss.detach().cpu().numpy()) < best_loss):
                                best_epoch = epoch
                                best_loss = np.min(loss.detach().cpu().numpy())

                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()

                    upsample = nn.Upsample(size=(img_size, img_size), mode='bilinear', align_corners=True)
                    real_mask = upsample(d_mask.module.mask.unsqueeze(0)).squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
                    real_mask_max = np.max(real_mask)
                    real_mask_min = np.min(real_mask)
                    mask_01 = (real_mask - real_mask_min) / (real_mask_max - real_mask_min)
                    img = x[i].permute(1, 2, 0).detach().cpu().numpy()
                    mix_img = mask_01 * img
                    mix_img = (mix_img - np.min(mix_img)) / (np.max(mix_img) - np.min(mix_img))
                    heatmap = cv2.applyColorMap(np.uint8(255 * mask_01), cv2.COLORMAP_JET)
                    heatmap = np.float32(heatmap) / 255
                    heatmap = heatmap[..., ::-1]
                    overlayed_original_img_j = 0.5 * img + 0.3 * heatmap
                    overlayed_original_img_j = (overlayed_original_img_j - np.min(overlayed_original_img_j)) / (np.max(overlayed_original_img_j) - np.min(overlayed_original_img_j))
                    plt.imsave(os.path.join(mask_img_path, str(base_mask_height_width_list[base_mask_num]) + 'x' + 'binary_mask' + '.png'), real_mask[:, :, 0])
                    plt.imsave(os.path.join(mask_img_path, str(base_mask_height_width_list[base_mask_num]) + 'x' + 'binary_mask_original' + '.png'), mix_img)
                    plt.imsave(os.path.join(mask_img_path, str(base_mask_height_width_list[base_mask_num]) + 'x' + 'heatmap_original' + '.png'), overlayed_original_img_j)

if __name__ == '__main__':
    # path of pre-trained model
    load_model_path = './pretrained_models/'
    ppnet = torch.load(load_model_path)
    ppnet = ppnet.cuda()
    ppnet_multi = torch.nn.DataParallel(ppnet)
    Dynamic_Mask_bb(ppnet)
