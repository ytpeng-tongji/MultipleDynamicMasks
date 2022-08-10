import numpy as np
import torch.nn as nn
import torch
from analysis_settings import base_mask_size_, base_mask_height_width_

class Dynamic_MaskPair(nn.Module):

    def __init__(self, img_size=224, mask_size=224, upper_y=0, lower_y=0, upper_x=0, lower_x=0, base_mask_size=base_mask_size_, base_mask_height_width=base_mask_height_width_):
        super(Dynamic_MaskPair, self).__init__()
        self.episilon = 1e-12
        self.init_weights = 0.5
        self.img_size = img_size
        self.mask_size = mask_size
        self.upper_y = upper_y
        self.lower_y = lower_y
        self.upper_x = upper_x
        self.lower_x = lower_x
        self.base_mask_size = base_mask_size
        self.base_mask_height = base_mask_height_width[0]
        self.base_mask_width = base_mask_height_width[1]
        self.mask_batch = int((self.img_size/self.mask_size) * (self.img_size/self.mask_size))
        init_mask = np.full((1, self.base_mask_size, self.base_mask_size), self.init_weights)
        init_mask = init_mask.astype('float32')
        self.mask = nn.Parameter(torch.tensor(init_mask), requires_grad=True)

    def forward(self, x):
        upsample = nn.Upsample(size=(self.img_size, self.img_size), mode='bilinear', align_corners=True)
        real_mask = upsample(self.mask.unsqueeze(0))
        x = real_mask * x

        return x