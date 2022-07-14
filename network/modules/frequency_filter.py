import os

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from einops import rearrange

from network import helper
from network.helper import tensor2img


class Filter(nn.Module):
    def __init__(self, size, band_start, band_end, use_learnable=True, norm=False):
        super(Filter, self).__init__()
        self.use_learnable = use_learnable

        self.base = nn.Parameter(torch.tensor(generate_filter(band_start, band_end, size)), requires_grad=False)
        if self.use_learnable:
            self.learnable = nn.Parameter(torch.randn(size, size), requires_grad=True)
            self.learnable.data.normal_(0., 0.1)
            # Todo
            # self.learnable = nn.Parameter(torch.rand((size, size)) * 0.2 - 0.1, requires_grad=True)

        self.norm = norm
        if norm:
            self.ft_num = nn.Parameter(torch.sum(torch.tensor(generate_filter(band_start, band_end, size))),
                                       requires_grad=False)

    def forward(self, x):
        if self.use_learnable:
            filt = self.base + norm_sigma(self.learnable)
        else:
            filt = self.base

        if self.norm:
            y = x * filt / self.ft_num
        else:
            y = x * filt
        return y


class FrequencyFilter(nn.Module):
    def __init__(self, image_size):
        super(FrequencyFilter, self).__init__()

        # init DCT matrix
        self._DCT_all = nn.Parameter(torch.tensor(DCT_mat(image_size)).float(), requires_grad=False)
        self._DCT_all_T = nn.Parameter(torch.transpose(torch.tensor(DCT_mat(image_size)).float(), 0, 1),
                                       requires_grad=False)

        # define base filters and learnable
        # 0 - 1/16 || 1/16 - 1/8 || 1/8 - 1
        low_filter = Filter(image_size, 0, image_size // 16)
        middle_filter = Filter(image_size, image_size // 16, image_size // 8)
        high_filter = Filter(image_size, image_size // 8, image_size)
        all_filter = Filter(image_size, 0, image_size * 2)

        self.filters = nn.ModuleList([low_filter, middle_filter, high_filter, all_filter])

    def forward(self, x):
        # DCT
        x_freq = self._DCT_all @ x @ self._DCT_all_T  # [N, 3, 299, 299]

        # 4 kernel
        y_list = []
        for i in range(len(self.filters)):
            x_pass = self.filters[i](x_freq)  # [N, 3, 299, 299]
            y = self._DCT_all_T @ x_pass @ self._DCT_all  # [N, 3, 299, 299]
            y_list.append(y)
        out = torch.cat(y_list, dim=1)  # [N, 12, 299, 299]
        return out


def DCT_mat(size):
    m = [[(np.sqrt(1. / size) if i == 0 else np.sqrt(2. / size)) * np.cos((j + 0.5) * np.pi * i / size) for j in
          range(size)] for i in range(size)]
    return m


def generate_filter(start, end, size):
    return [[0. if i + j > end or i + j <= start else 1. for j in range(size)] for i in range(size)]


def norm_sigma(x):
    return 2. * torch.sigmoid(x) - 1.


if __name__ == '__main__':
    path = r'D:\test'
    out_path = r'D:\images'
    for f in os.listdir(path):
        img = Image.open(os.path.join(path, f))
        x = helper.torch_resize(img)
        model = FrequencyFilter(224)
        out = model(x)
        out = rearrange(out, 'b (t c) h w -> b t c h w', c=3)
        for i in range(out.size(1)):
            image = tensor2img(out[0][i])
            image.save(os.path.join(out_path, f'{i}-{f}'))
