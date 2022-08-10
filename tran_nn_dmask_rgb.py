from analysis_settings import img_size_, original_img_path_, mean, std, dynamic_img_batch_, dynamic_mask_lrs,\
best_loss_, best_epoch_, iteration_epoch_, iteration_epoch_min_, patient_, mask_optimizer_lr_, check_epoch_, base_mask_height_width_list_
import torch.utils.model_zoo as model_zoo
import os
import torch
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch import nn
from utils.helpers import makedir
import dynamic_maskpair
import numpy as np
import cv2
import matplotlib.pyplot as plt
import model # pre-trained model architecture

num_classes = 200
episilon = 1e-12
original_img_path = original_img_path_
img_size = img_size_
dynamic_img_batch = dynamic_img_batch_
img_mask_root_path = './img_mask_root_path'
normalize = transforms.Normalize(mean=mean, std=std)

def Dynamic_Mask_bb(model):
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

    for batch_idx, (x, _) in enumerate(original_img_loader):
        x = x.cuda()
        output = model(x)
        predicted, predicted_index = torch.max(output.data, 1)

        for i in range(x.shape[0]):
            base_mask_height_width_list = base_mask_height_width_list_

            for base_mask_num in range(len(base_mask_height_width_list)):
                mask_img_path = os.path.join(img_mask_root_path, str(batch_idx * x.shape[0] + i))
                makedir(mask_img_path)
                d_mask = dynamic_maskpair.Dynamic_MaskPair(img_size=img_size, base_mask_height_width=base_mask_height_width_list[base_mask_num])
                d_mask = d_mask.cuda()
                original_max = predicted[i]
                max_activation = original_max.cuda()
                max_activation_index = predicted_index[i]
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
                    if ((epoch - best_epoch) > patient and epoch > iteration_epoch_min):
                        break
                    else:
                        x_mask = d_mask(x[i])
                        torch_x_mask = model(x_mask)
                        torch_x_mask_activation = torch_x_mask[0][max_activation_index].cuda()
                        mse_loss = (torch_x_mask_activation - max_activation)**2
                        mse_loss = mse_loss.cuda()
                        mean_l1_loss = d_mask.module.mask.norm(p=1, dim=(1, 2)) / (base_mask_height * base_mask_width)
                        loss = dynamic_mask_lrs['mse'] * mse_loss + dynamic_mask_lrs['l1'] * mean_l1_loss
                        loss = loss.sum()
                        loss = loss.cuda()

                        if (epoch % check_epoch == 0):
                            print("epoch = ", epoch)
                            print("mse_loss = ", mse_loss)
                            print("l1_loss = ", mean_l1_loss)
                            print("loss = ", loss)

                        if (np.min(loss.detach().cpu().numpy()) < best_loss):
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
    resnet = torch.load(load_model_path)
    model = resnet.cuda()
    model_multi = torch.nn.DataParallel(model)
    Dynamic_Mask_bb(model_multi)

