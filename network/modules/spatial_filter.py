import os

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from einops import rearrange
from torch.nn import functional as F

from network import helper
from network.helper import tensor2img


def get_SRM_op():
    r = np.array([
        [0, 0, 0, 0, 0],
        [0, -1, 2, -1, 0],
        [0, 2, -4, 2, 0],
        [0, -1, 2, -1, 0],
        [0, 0, 0, 0, 0]
    ]).astype(np.float32) / 8
    g = np.array([
        [-1, 2, -2, 2, -1],
        [2, -6, 8, -6, 2],
        [-2, 8, -12, 8, -2],
        [2, -6, 8, -6, 2],
        [-1, 2, -2, 2, -1],
    ]).astype(np.float32) / 8
    b = np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 1, -2, 1, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ]).astype(np.float32) / 8

    return r, g, b


def get_laplace_op():
    laplace_operator_x = np.array([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0],
    ]).astype(np.float32) / 8
    laplace_operator_y = np.array([
        [1, 1, 1],
        [1, -8, 1],
        [1, 1, 1],
    ]).astype(np.float32) / 8
    return laplace_operator_x, laplace_operator_y


def get_sobel_op():
    sobel_operator_x = np.array([
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1],
    ]).astype(np.float32) / 2
    sobel_operator_y = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1],
    ]).astype(np.float32) / 2
    return sobel_operator_x, sobel_operator_y


class SpatialFilter(nn.Module):
    sobel_operator_x, sobel_operator_y = get_sobel_op()
    laplace_operator_x, laplace_operator_y = get_laplace_op()
    srm_r, srm_g, srm_b = get_SRM_op()

    def __init__(self, out_size):
        super().__init__()
        self.sobel_x = self.build_conv_filter(self.sobel_operator_x)
        self.sobel_y = self.build_conv_filter(self.sobel_operator_y)
        self.laplacian_x = self.build_conv_filter(self.laplace_operator_x)
        self.laplacian_y = self.build_conv_filter(self.laplace_operator_y)
        self.srm_filter = self.build_srm_5x5_filter()
        self.out_size = out_size

    def build_conv_filter(self, x, size=3):
        x = x.reshape((1, 1, size, size))
        x = np.repeat(x, 3, axis=1)
        x = np.repeat(x, 3, axis=0)

        x = torch.from_numpy(x)
        x = nn.Parameter(x, requires_grad=False)
        conv_x = nn.Conv2d(3, 3, kernel_size=size, padding='same', bias=False)
        conv_x.weight = x
        return nn.Sequential(conv_x, nn.BatchNorm2d(3))

    def sobel_filter(self, x):
        return self.run_filter(x, self.sobel_x, self.sobel_y)

    def laplacian_filter(self, x):
        return self.run_filter(x, self.laplacian_x, self.laplacian_y)

    def run_filter(self, x, filter_x, filter_y):
        g_x = filter_x(x)
        g_y = filter_y(x)
        g = torch.sqrt(torch.pow(g_x, 2) + torch.pow(g_y, 2))
        x = torch.sigmoid(g) * x
        return x

    def build_srm_5x5_filter(self):
        srm = np.asarray([self.srm_r, self.srm_g, self.srm_b])
        srm = srm.reshape((1, 3, 5, 5))
        srm = np.repeat(srm, 3, axis=0)
        srm = torch.from_numpy(srm)
        srm = nn.Parameter(srm, requires_grad=False)
        conv_x = nn.Conv2d(3, 3, kernel_size=5, padding='same', bias=False)
        conv_x.weight = srm
        return nn.Sequential(conv_x, nn.BatchNorm2d(3))

    def resize(self, x):
        return F.interpolate(x, size=self.out_size, mode='bilinear', align_corners=True)

    def forward(self, x):
        x1 = self.sobel_filter(x)
        x1 = self.resize(x1)
        x2 = self.laplacian_filter(x)
        x2 = self.resize(x2)
        x3 = self.srm_filter(x)
        x3 = self.resize(x3)
        x = self.resize(x)
        return torch.cat([x, x1, x2, x3], dim=1)  # (b, 12, h, w)


if __name__ == '__main__':
    path = r'D:\test'
    out_path = r'D:\tmp'
    for f in os.listdir(path):
        img = Image.open(os.path.join(path, f))
        x = helper.torch_resize(img)
        model = SpatialFilter(224)
        out = model(x)
        out = rearrange(out, 'b (t c) h w -> b t c h w', c=3)
        for i in range(out.size(1)):
            image = tensor2img(out[0][i])
            image.save(os.path.join(out_path, f'{i}-{f}'))
